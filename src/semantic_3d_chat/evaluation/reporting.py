"""Artifact-only research report and figure generation.

This module never loads model weights or performs inference.  It summarizes
JSON/JSONL/NPZ-adjacent metadata already produced by the pipeline, and marks
every absent experiment explicitly instead of filling gaps with estimates.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from semantic_3d_chat.mapping.voxel_map import PERSISTED_MAP_CONTENT_HASH_DOMAIN


@dataclass(frozen=True)
class ReportInputs:
    project_root: Path
    reports_root: Path
    scene_id: str
    metrics: dict[str, dict[str, Any] | None]
    sources: dict[str, str]
    missing: tuple[str, ...]
    warnings: tuple[str, ...]
    checkpoint_history: tuple[dict[str, Any], ...]
    best_checkpoint: dict[str, Any] | None
    qa_counts: dict[str, int]
    qa_splits: dict[str, Any] | None
    render_frame_count: int | None
    sample_chats: tuple[dict[str, Any], ...]
    training_namespace: str | None


def _configured_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _artifact_root(root: Path, config: Mapping[str, Any], kind: str) -> Path:
    paths = config["paths"]
    value = paths.get(f"{kind}_root")
    if value is None:
        value = Path(str(paths.get("data_root", "data"))) / kind
    return _configured_path(root, value)


def _load_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        warnings.append(f"Could not read {path}: {type(error).__name__}: {error}")
        return None
    if not isinstance(value, dict):
        warnings.append(f"Expected a JSON object in {path}")
        return None
    return value


def _load_jsonl(path: Path, warnings: list[str], limit: int = 20) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not an object")
            records.append(value)
            if len(records) >= limit:
                break
    except (OSError, json.JSONDecodeError, TypeError) as error:
        warnings.append(f"Could not read {path}: {type(error).__name__}: {error}")
    return tuple(records)


def _persisted_map_hash_pair(
    map_metrics: Mapping[str, Any],
    semantic_metrics: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Return only hashes known to share the persisted-array domain.

    Legacy mapping reports used ``content_hash`` for persisted numeric arrays
    without naming the domain, so that field remains comparable. Legacy
    semantic reports used ``map_content_hash`` after reconstructing the map;
    an undeclared semantic hash is therefore intentionally not compared.
    """

    mapping_domain = map_metrics.get("content_hash_domain")
    if mapping_domain not in (None, PERSISTED_MAP_CONTENT_HASH_DOMAIN):
        return None
    mapping_hash = map_metrics.get("content_hash")
    if not isinstance(mapping_hash, str) or not mapping_hash:
        return None

    semantic_hash = semantic_metrics.get("map_persisted_content_hash")
    if semantic_hash is not None:
        semantic_domain = semantic_metrics.get("map_persisted_content_hash_domain")
        if semantic_domain not in (None, PERSISTED_MAP_CONTENT_HASH_DOMAIN):
            return None
    elif semantic_metrics.get("map_content_hash_domain") == PERSISTED_MAP_CONTENT_HASH_DOMAIN:
        semantic_hash = semantic_metrics.get("map_content_hash")
    else:
        return None
    if not isinstance(semantic_hash, str) or not semantic_hash:
        return None
    return mapping_hash, semantic_hash


def _metric_candidates(scene_id: str) -> dict[str, tuple[str, ...]]:
    return {
        "machine": ("machine_report.json",),
        "models": ("model_revisions.json",),
        "geometry": ("geometry.json",),
        "map": (f"map_{scene_id}.json", "map.json"),
        "semantic": (f"semantic_sanity_{scene_id}.json", "semantic_sanity.json"),
        "qa": ("metrics.json", "evaluation.json", "qa.json"),
        "leakage": ("leakage.json",),
        "ablations": ("ablations.json",),
        "baselines": ("baselines.json",),
        "robot": ("robot.json", "robot_navigation.json"),
    }


def _optional_metric_candidates() -> dict[str, tuple[str, ...]]:
    """Return historical/diagnostic inputs that are not required report groups."""

    return {
        "training_v1": ("training_multiscene.json",),
        "training_v2": ("training_multiscene_anticollapse.json",),
        "qa_v1": ("metrics_v1_collapsed.json", "metrics_multiscene_v1.json"),
        "qa_v2": ("metrics_v2_anticollapse.json", "metrics_anticollapse.json"),
        "validation": ("validation_metrics.json",),
        "signal_audit": ("scene_signal_audit.json",),
        "resampler_diagnostic": ("resampler_fix_diagnostic.json",),
        "direct_multiview": ("direct_multiview.json",),
        "oracle_text": ("oracle_text.json",),
    }


def _training_namespaces(config: Mapping[str, Any]) -> tuple[str | None, ...]:
    configured = config.get("training", {}).get("output_namespace")
    candidates: list[str | None] = []
    preferred = (
        (configured, "multiscene_anticollapse", "multiscene", None)
        if configured
        else ("multiscene_anticollapse", "multiscene", None)
    )
    for value in preferred:
        normalized = str(value) if value else None
        if normalized not in candidates:
            candidates.append(normalized)
    return tuple(candidates)


def _select_training_namespace(
    root: Path,
    reports_root: Path,
    config: Mapping[str, Any],
) -> str | None:
    """Select the most relevant completed/in-progress run without loading weights."""

    metrics_root = reports_root / "metrics"
    for namespace in _training_namespaces(config):
        metrics_name = "training.json" if namespace is None else f"training_{namespace}.json"
        checkpoint_root = _artifact_root(root, config, "checkpoints")
        if namespace is not None:
            checkpoint_root = checkpoint_root / namespace
        if (metrics_root / metrics_name).is_file() or (
            checkpoint_root / "best/metadata.json"
        ).is_file():
            return namespace
    return None


def collect_report_inputs(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    scene_id: str | None = None,
) -> ReportInputs:
    root = Path(project_root).resolve()
    reports_root = _configured_path(
        root, config["paths"].get("reports_root", "reports")
    )
    metrics_root = reports_root / "metrics"
    selected_scene = scene_id or str(config.get("scene", {}).get("scene_id", "scene_000001"))
    warnings: list[str] = []
    metrics: dict[str, dict[str, Any] | None] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for name, candidates in _metric_candidates(selected_scene).items():
        selected_path = next(
            (
                metrics_root / candidate
                for candidate in candidates
                if (metrics_root / candidate).is_file()
            ),
            None,
        )
        if selected_path is None:
            metrics[name] = None
            missing.append(name)
            continue
        metrics[name] = _load_json(selected_path, warnings)
        sources[name] = str(selected_path)
        if metrics[name] is None:
            missing.append(name)

    for name, candidates in _optional_metric_candidates().items():
        selected_path = next(
            (
                metrics_root / candidate
                for candidate in candidates
                if (metrics_root / candidate).is_file()
            ),
            None,
        )
        metrics[name] = None if selected_path is None else _load_json(selected_path, warnings)
        if selected_path is not None:
            sources[name] = str(selected_path)
    if metrics.get("baselines") is None and (
        metrics.get("direct_multiview") is not None or metrics.get("oracle_text") is not None
    ):
        metrics["baselines"] = {
            "direct_multiview": metrics.get("direct_multiview"),
            "oracle_text": metrics.get("oracle_text"),
        }
        if metrics.get("direct_multiview") is not None and metrics.get("oracle_text") is not None:
            missing = [name for name in missing if name != "baselines"]

    training_namespace = _select_training_namespace(root, reports_root, config)
    training_metrics_name = (
        "training.json" if training_namespace is None else f"training_{training_namespace}.json"
    )
    training_metrics_path = metrics_root / training_metrics_name
    metrics["training"] = _load_json(training_metrics_path, warnings)
    if metrics["training"] is None:
        missing.append("training")
    else:
        sources["training"] = str(training_metrics_path)

    checkpoints_root = _artifact_root(root, config, "checkpoints")
    if training_namespace is not None:
        checkpoints_root = checkpoints_root / training_namespace
    checkpoint_history: list[dict[str, Any]] = []
    for path in sorted(checkpoints_root.glob("epoch_*/metadata.json")):
        metadata = _load_json(path, warnings)
        if metadata is not None and isinstance(metadata.get("epoch"), int):
            checkpoint_history.append({**metadata, "metadata_path": str(path)})
    checkpoint_history.sort(key=lambda record: int(record["epoch"]))
    best_checkpoint = _load_json(checkpoints_root / "best" / "metadata.json", warnings)
    if best_checkpoint is not None:
        sources["best_checkpoint"] = str(checkpoints_root / "best" / "metadata.json")

    qa_root = _artifact_root(root, config, "qa")
    qa_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = qa_root / f"{split}.jsonl"
        if path.is_file():
            qa_counts[split] = sum(
                1 for line in path.read_text(encoding="utf-8").splitlines() if line
            )
        else:
            qa_counts[split] = 0
    qa_splits = _load_json(qa_root / "splits.json", warnings)

    render_manifest = _artifact_root(root, config, "rendered") / selected_scene / "manifest.json"
    render_payload = _load_json(render_manifest, warnings)
    render_frame_count = None
    if render_payload is not None and isinstance(render_payload.get("frames"), list):
        render_frame_count = len(render_payload["frames"])
        sources["render_manifest"] = str(render_manifest)
    sample_chats = _load_jsonl(reports_root / "examples" / "sample_chats.jsonl", warnings)
    return ReportInputs(
        project_root=root,
        reports_root=reports_root,
        scene_id=selected_scene,
        metrics=metrics,
        sources=sources,
        missing=tuple(sorted(set(missing))),
        warnings=tuple(warnings),
        checkpoint_history=tuple(checkpoint_history),
        best_checkpoint=best_checkpoint,
        qa_counts=qa_counts,
        qa_splits=qa_splits,
        render_frame_count=render_frame_count,
        sample_chats=sample_chats,
        training_namespace=training_namespace,
    )


def _new_figure(width: float, height: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure


def architecture_figure(path: Path, config: Mapping[str, Any]) -> Path:
    """Draw the primary no-text-environment data path and oracle boundary."""

    from matplotlib.patches import FancyBboxPatch

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = _new_figure(13.5, 6.2)
    axes = figure.add_subplot(1, 1, 1)
    axes.set_xlim(0, 14)
    axes.set_ylim(0, 7)
    axes.axis("off")
    vision = config["vision"]
    mapping = config["mapping"]
    scene_encoder = config["scene_encoder"]
    language = config["language"]
    architecture_version = str(scene_encoder.get("architecture_version", "legacy_perceiver_v1"))
    global_encoder_label = (
        "Spatial coverage\nresampler"
        if architecture_version == "spatial_coverage_resampler_v2"
        else "Global Perceiver"
    )
    boxes = [
        (0.3, 4.4, 1.55, 1.25, "RGB + metric depth\nexact pose/intrinsics", "#dbeafe"),
        (
            2.2,
            4.4,
            1.75,
            1.25,
            f"Full-image CLIP\none pass per view\n{vision.get('aligned_method', 'aligned')} tokens",
            "#ede9fe",
        ),
        (
            4.3,
            4.4,
            1.65,
            1.25,
            f"3D voxel fusion\n{mapping['voxel_size_m']:.2f} m voxels\n2,048-D float16",
            "#dcfce7",
        ),
        (
            6.3,
            4.4,
            1.7,
            1.25,
            f"All spatial blocks\n{scene_encoder['block_size_m']:.2f} m blocks\nno retrieval",
            "#fef3c7",
        ),
        (
            8.35,
            4.4,
            1.7,
            1.25,
            f"{global_encoder_label}\n{scene_encoder['global_latents']} latents\n{scene_encoder['model_dim']}-D",
            "#ffedd5",
        ),
        (
            10.4,
            4.4,
            1.55,
            1.25,
            "Continuous prefix\ninputs_embeds\nquestion-independent",
            "#fee2e2",
        ),
        (
            12.3,
            4.4,
            1.4,
            1.25,
            f"Local LM\n{language['model_id'].split('/')[-1]}\nanswer",
            "#e0f2fe",
        ),
    ]
    for x, y, width, height, label, color in boxes:
        axes.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="#334155",
                linewidth=1.2,
            )
        )
        axes.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8.5)
    for left, right in pairwise(boxes):
        axes.annotate(
            "",
            xy=(right[0] - 0.06, right[1] + right[3] / 2),
            xytext=(left[0] + left[2] + 0.06, left[1] + left[3] / 2),
            arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.3},
        )
    axes.add_patch(
        FancyBboxPatch(
            (2.6, 1.0),
            5.0,
            1.25,
            boxstyle="round,pad=0.06",
            facecolor="#fafafa",
            edgecolor="#b91c1c",
            linestyle="--",
            linewidth=1.4,
        )
    )
    axes.text(
        5.1,
        1.63,
        "Oracle / QA metadata — evaluation and supervision only\nnever opened by primary chat runtime",
        ha="center",
        va="center",
        fontsize=9,
        color="#991b1b",
    )
    axes.annotate(
        "supervision / scoring only",
        xy=(11.1, 4.35),
        xytext=(8.2, 2.75),
        ha="center",
        fontsize=8,
        color="#991b1b",
        arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "#b91c1c"},
    )
    axes.text(
        7,
        6.45,
        "Semantic 3D Chat — primary continuous environment path",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    axes.text(
        7,
        0.35,
        "User questions are text; environmental evidence remains continuous features + geometry.",
        ha="center",
        fontsize=9,
        color="#334155",
    )
    figure.savefig(path, format="png")
    figure.clear()
    return path


def training_loss_figure(path: Path, history: Sequence[Mapping[str, Any]]) -> Path | None:
    rows = [
        (
            int(record["epoch"]),
            float(record["train_loss"]),
            (
                float(record["validation_loss"])
                if record.get("validation_loss") is not None
                else None
            ),
        )
        for record in history
        if record.get("epoch") is not None and record.get("train_loss") is not None
    ]
    if not rows:
        return None
    rows.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = _new_figure(7.2, 4.5)
    axes = figure.add_subplot(1, 1, 1)
    axes.plot(
        [row[0] for row in rows],
        [row[1] for row in rows],
        marker="o",
        linewidth=2,
        label="train",
    )
    validation_rows = [row for row in rows if row[2] is not None]
    if validation_rows:
        axes.plot(
            [row[0] for row in validation_rows],
            [row[2] for row in validation_rows],
            marker="o",
            linewidth=2,
            label="validation",
        )
        axes.legend()
    axes.set(xlabel="Epoch", ylabel="Cross-entropy", title="Selected adapter run")
    axes.grid(alpha=0.25)
    axes.set_xticks([row[0] for row in rows])
    figure.savefig(path, format="png")
    figure.clear()
    return path


def semantic_category_figure(path: Path, semantic: Mapping[str, Any] | None) -> Path | None:
    if not semantic or not isinstance(semantic.get("queries"), list):
        return None
    rows = [record for record in semantic["queries"] if isinstance(record, Mapping)]
    if not rows:
        return None
    labels = [str(record.get("category", record.get("query_id", "?"))) for record in rows]
    precision = [float(record.get("precision_at_k") or 0.0) for record in rows]
    chance = [float(record.get("random_precision_at_k") or 0.0) for record in rows]
    hit = [float(bool(record.get("hit_at_k"))) for record in rows]
    y = np.arange(len(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = _new_figure(9.2, max(4.8, 0.42 * len(rows) + 1.8))
    axes = figure.add_subplot(1, 1, 1)
    axes.barh(y - 0.2, precision, height=0.2, label="precision@k")
    axes.barh(y, chance, height=0.2, label="random precision")
    axes.barh(y + 0.2, hit, height=0.2, label="hit@k")
    axes.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Score",
        xlim=(0, 1),
        title="Zero-shot 3D localization by semantic category",
    )
    axes.invert_yaxis()
    axes.grid(axis="x", alpha=0.2)
    axes.legend(loc="lower right")
    figure.savefig(path, format="png")
    figure.clear()
    return path


def accuracy_by_type_figure(path: Path, qa_metrics: Mapping[str, Any] | None) -> Path | None:
    per_type = qa_metrics.get("per_type") if qa_metrics else None
    if not isinstance(per_type, Mapping) or not per_type:
        return None
    rows = [
        (str(name), values.get("normalized_exact_accuracy"))
        for name, values in sorted(per_type.items())
        if isinstance(values, Mapping) and values.get("normalized_exact_accuracy") is not None
    ]
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = _new_figure(8.0, 4.8)
    axes = figure.add_subplot(1, 1, 1)
    axes.bar([row[0] for row in rows], [float(row[1]) for row in rows], color="#2563eb")
    axes.set(ylabel="Normalized exact accuracy", ylim=(0, 1), title="Held-out QA by type")
    axes.tick_params(axis="x", rotation=35)
    axes.grid(axis="y", alpha=0.2)
    figure.savefig(path, format="png")
    figure.clear()
    return path


def counterfactual_figure(path: Path, qa_metrics: Mapping[str, Any] | None) -> Path | None:
    values = qa_metrics.get("counterfactual") if qa_metrics else None
    if not isinstance(values, Mapping) or not values.get("eligible_pairs"):
        return None
    keys = (
        ("Pair accuracy", "pair_accuracy"),
        ("Changed when expected", "changed_when_expected_rate"),
        ("Invariant when expected", "invariant_when_expected_rate"),
    )
    rows = [(label, values.get(key)) for label, key in keys if values.get(key) is not None]
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = _new_figure(7.4, 4.4)
    axes = figure.add_subplot(1, 1, 1)
    axes.bar([row[0] for row in rows], [float(row[1]) for row in rows], color="#7c3aed")
    axes.set(ylabel="Rate", ylim=(0, 1), title="Counterfactual-pair consistency")
    axes.tick_params(axis="x", rotation=20)
    axes.grid(axis="y", alpha=0.2)
    figure.savefig(path, format="png")
    figure.clear()
    return path


def _ablation_rows(payload: Mapping[str, Any] | None) -> list[tuple[str, float]]:
    if not payload:
        return []
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    raw = payload.get("results", payload.get("ablations"))
    if isinstance(raw, Mapping):
        candidates = [
            (str(name), value) for name, value in raw.items() if isinstance(value, Mapping)
        ]
    elif isinstance(raw, list):
        candidates = [
            (
                str(value.get("mode", value.get("name", f"ablation_{index}"))),
                value.get("metrics", value),
            )
            for index, value in enumerate(raw)
            if isinstance(value, Mapping)
        ]
    rows: list[tuple[str, float]] = []
    for name, metrics in candidates:
        if not isinstance(metrics, Mapping):
            continue
        value = next(
            (
                metrics[key]
                for key in (
                    "normalized_exact_accuracy",
                    "accuracy",
                    "top_k_localization_accuracy",
                )
                if isinstance(metrics.get(key), (int, float))
            ),
            None,
        )
        if value is not None:
            rows.append((name, float(value)))
    return rows


def ablation_figure(path: Path, payload: Mapping[str, Any] | None) -> Path | None:
    rows = _ablation_rows(payload)
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = _new_figure(8.2, 4.8)
    axes = figure.add_subplot(1, 1, 1)
    axes.bar([row[0] for row in rows], [row[1] for row in rows], color="#dc2626")
    axes.set(ylabel="Primary score", ylim=(0, 1), title="Ablation results")
    axes.tick_params(axis="x", rotation=35)
    axes.grid(axis="y", alpha=0.2)
    figure.savefig(path, format="png")
    figure.clear()
    return path


def scan_montage_figure(path: Path, inputs: ReportInputs, config: Mapping[str, Any]) -> Path | None:
    rendered_root = _artifact_root(inputs.project_root, config, "rendered")
    images = sorted((rendered_root / inputs.scene_id / "rgb").glob("*.png"))
    if not images:
        return None
    selected_count = min(12, len(images))
    indices = np.linspace(0, len(images) - 1, selected_count, dtype=np.int64)
    selected = [images[index] for index in indices]
    columns = 4
    rows = math.ceil(len(selected) / columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = _new_figure(10.5, 2.55 * rows)
    from PIL import Image

    for index, image_path in enumerate(selected):
        axes = figure.add_subplot(rows, columns, index + 1)
        with Image.open(image_path) as image:
            axes.imshow(image.convert("RGB"))
        axes.set_title(image_path.stem, fontsize=8)
        axes.axis("off")
    for index in range(len(selected), rows * columns):
        axes = figure.add_subplot(rows, columns, index + 1)
        axes.axis("off")
    figure.suptitle(f"Deterministic center scan — {len(images)} total RGB-D poses", fontsize=13)
    figure.savefig(path, format="png")
    figure.clear()
    return path


def generate_report_figures(inputs: ReportInputs, config: Mapping[str, Any]) -> dict[str, str]:
    figures_root = inputs.reports_root / "figures"
    outputs: dict[str, str] = {}

    def record(name: str, path: Path | None) -> None:
        if path is not None:
            outputs[name] = str(path.relative_to(inputs.reports_root))

    record("architecture", architecture_figure(figures_root / "architecture.png", config))
    record(
        "scan_montage",
        scan_montage_figure(figures_root / "scan_montage.png", inputs, config),
    )
    record(
        "training_loss",
        training_loss_figure(
            figures_root / "training_loss.png",
            (
                inputs.metrics["training"].get("history", inputs.checkpoint_history)
                if inputs.metrics["training"]
                else inputs.checkpoint_history
            ),
        ),
    )
    record(
        "semantic_by_category",
        semantic_category_figure(
            figures_root / "semantic_localization_by_category.png", inputs.metrics["semantic"]
        ),
    )
    record(
        "accuracy_by_type",
        accuracy_by_type_figure(
            figures_root / "accuracy_by_question_type.png", inputs.metrics["qa"]
        ),
    )
    record(
        "counterfactual",
        counterfactual_figure(
            figures_root / "counterfactual_consistency.png", inputs.metrics["qa"]
        ),
    )
    record(
        "ablations",
        ablation_figure(figures_root / "ablation_accuracy.png", inputs.metrics["ablations"]),
    )
    return outputs


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "Not measured"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "Not measured"
        return f"{value:.{digits}f}"
    return str(value)


def _percent(value: Any, digits: int = 1) -> str:
    return "Not measured" if value is None else f"{float(value) * 100:.{digits}f}%"


def _gib(value: Any) -> str:
    return "Not measured" if not isinstance(value, (int, float)) else f"{value / 2**30:.1f} GiB"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(_escape_cell(value) for value in row) + " |" for row in rows)
    return result


def _figure_markdown(figures: Mapping[str, str], key: str, alt: str) -> str:
    path = figures.get(key)
    return (
        f"![{alt}]({path})"
        if path
        else f"*{alt}: not generated because source measurements are absent.*"
    )


def _qa_split_summary(inputs: ReportInputs) -> str:
    return ", ".join(f"{name}={count}" for name, count in inputs.qa_counts.items())


def _latest_training(inputs: ReportInputs) -> tuple[int | None, float | None]:
    if inputs.checkpoint_history:
        latest = inputs.checkpoint_history[-1]
        return int(latest["epoch"]), float(latest["train_loss"])
    training = inputs.metrics.get("training")
    if training:
        return int(training.get("epochs", 0)) or None, training.get("best_train_loss")
    return None, None


def _qa_lineage(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return "unmeasured"
    source = str(payload.get("predictions_path", "")).lower()
    if "anticollapse" in source or "v2" in source:
        return "v2"
    if "multiscene" in source:
        return "v1"
    return "unversioned"


def _has_heldout_qa(inputs: ReportInputs, qa: Mapping[str, Any] | None) -> bool:
    return bool(
        qa and int(qa.get("reference_count", 0)) > 0 and inputs.qa_counts.get("test", 0) > 0
    )


def _v1_collapse_evidence(inputs: ReportInputs) -> bool:
    audit = inputs.metrics.get("signal_audit") or {}
    corroborating = audit.get("corroborating_control_results", {})
    if corroborating and float(corroborating.get("primary_changed_when_expected_rate", 1.0)) == 0:
        return True
    interpretation = (inputs.metrics.get("ablations") or {}).get("interpretation", {})
    warning = str(interpretation.get("warning", "")).lower()
    return bool(
        "first multiscene checkpoint" in warning
        and "insensitive" in warning
        and interpretation.get("empty_prefix_collapse")
    )


def _control_table_rows(payload: Mapping[str, Any] | None) -> list[tuple[Any, ...]]:
    if not payload:
        return []
    raw = payload.get("results")
    if not isinstance(raw, Mapping):
        return []
    preferred = (
        "primary",
        "wrong_scene_prefix",
        "semantic_shuffle",
        "position_shuffle",
        "geometry_only",
        "semantics_without_xyz",
        "remove_rgb",
        "remove_normals",
        "empty_scene_prefix",
    )
    ordered = [name for name in preferred if name in raw]
    ordered.extend(sorted(str(name) for name in raw if name not in ordered))
    rows: list[tuple[Any, ...]] = []
    for name in ordered:
        values = raw.get(name)
        if not isinstance(values, Mapping):
            continue
        rows.append(
            (
                name,
                _percent(values.get("normalized_exact_accuracy")),
                _percent(values.get("spatial_relation_accuracy")),
                _percent(values.get("counterfactual_changed_rate")),
                _format_number(values.get("exact_accuracy_delta_vs_primary"), 4),
            )
        )
    return rows


def _baseline_payload(
    inputs: ReportInputs,
    name: str,
) -> Mapping[str, Any] | None:
    aggregate = inputs.metrics.get("baselines") or {}
    value = aggregate.get(name)
    if isinstance(value, Mapping):
        return value
    standalone = inputs.metrics.get(name)
    return standalone if isinstance(standalone, Mapping) else None


def _json_block(payload: Mapping[str, Any], limit: int = 4000) -> str:
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True)[:limit] + "\n```"


def render_final_report(
    inputs: ReportInputs,
    config: Mapping[str, Any],
    figures: Mapping[str, str],
) -> str:
    machine = inputs.metrics["machine"] or {}
    models = inputs.metrics["models"] or {}
    geometry = inputs.metrics["geometry"]
    map_metrics = inputs.metrics["map"]
    semantic = inputs.metrics["semantic"]
    training = inputs.metrics["training"]
    qa = inputs.metrics["qa"]
    leakage = inputs.metrics["leakage"]
    ablations = inputs.metrics["ablations"]
    robot = inputs.metrics["robot"]
    training_v1 = inputs.metrics.get("training_v1")
    training_v2 = inputs.metrics.get("training_v2")
    qa_v1 = inputs.metrics.get("qa_v1")
    qa_v2 = inputs.metrics.get("qa_v2")
    signal_audit = inputs.metrics.get("signal_audit")
    resampler_diagnostic = inputs.metrics.get("resampler_diagnostic")
    checkpoint = inputs.best_checkpoint or {}
    latest_epoch, latest_loss = _latest_training(inputs)
    semantic_aggregate = semantic.get("aggregate", {}) if semantic else {}
    consistency = semantic.get("same_voxel_consistency", {}) if semantic else {}
    qa_grounding = qa.get("grounding", {}) if qa else {}
    counterfactual = qa.get("counterfactual", {}) if qa else {}
    startup = leakage.get("startup", {}) if leakage else {}
    generated = datetime.now(UTC).replace(microsecond=0).isoformat()
    qa_lineage = _qa_lineage(qa)
    if qa_v1 is None and qa_lineage == "v1":
        qa_v1 = qa
    if qa_v2 is None and qa_lineage == "v2":
        qa_v2 = qa
    heldout_measured = _has_heldout_qa(inputs, qa)
    v1_collapse = _v1_collapse_evidence(inputs)
    architecture_version = str(
        checkpoint.get(
            "scene_encoder_architecture_version",
            config["scene_encoder"].get("architecture_version", "legacy_perceiver_v1"),
        )
    )
    direct_baseline = _baseline_payload(inputs, "direct_multiview")
    oracle_baseline = _baseline_payload(inputs, "oracle_text")
    ablation_checkpoint = (ablations or {}).get("checkpoint")
    ablation_architecture = (ablations or {}).get("scene_encoder_architecture_version")
    ablation_provenance_recorded = bool(ablation_checkpoint or ablation_architecture)

    if heldout_measured:
        evidence_summary = (
            f"Held-out QA is measured on {qa.get('reference_count')} test records at "
            f"{_percent(qa.get('normalized_exact_accuracy'))} exact accuracy."
        )
        if qa_lineage == "v1" and v1_collapse:
            evidence_summary += (
                " The v1 controls invalidate that raw score as evidence of scene-specific "
                "understanding: wrong-scene and shuffled-content prefixes matched or exceeded "
                "the primary path, and changed-fact consistency was zero."
            )
        elif qa_lineage == "v2":
            evidence_summary += (
                " This is the v2 behavioral result; its content sensitivity must be judged "
                "from the paired and ablation controls below, not from accuracy alone."
            )
    else:
        evidence_summary = (
            "No held-out QA prediction artifact with a non-empty test split is available; "
            "the report therefore makes no generalization claim."
        )
    if resampler_diagnostic and qa_v2 is None:
        evidence_summary += (
            " The v2 resampler has a structural, no-training signal-preservation diagnostic, "
            "but no v2 held-out behavioral result is yet associated with the report artifacts."
        )

    split_values = (inputs.qa_splits or {}).get("splits", {})
    split_scene_count = sum(
        len(value) for value in split_values.values() if isinstance(value, list)
    )
    limitations: list[str] = []
    if split_scene_count <= 1:
        limitations.append("Only one scene is represented in the recorded scene split.")
    if inputs.qa_counts.get("validation", 0) == 0 or inputs.qa_counts.get("test", 0) == 0:
        limitations.append(
            "Validation or test records are absent; held-out generalization is not measured."
        )
    if v1_collapse:
        limitations.append(
            "The v1 multi-scene adapter is scene-content-insensitive despite its raw "
            "held-out accuracy; wrong-scene and content-shuffle controls invalidate a "
            "scene-understanding claim for that checkpoint."
        )
    if architecture_version == "spatial_coverage_resampler_v2" and qa_v2 is None:
        limitations.append(
            "The v2 structural diagnostic preserves more scene signal, but no explicitly "
            "v2-tagged held-out QA artifact is available yet."
        )
    if (training or {}).get("elapsed_seconds") is None:
        limitations.append("Selected-run wall-clock training time is not recorded.")
    limitations.append("Peak training memory is not recorded.")
    if not counterfactual.get("eligible_pairs"):
        limitations.append("Counterfactual QA is not scored.")
    elif float(counterfactual.get("changed_when_expected_rate", 0.0)) == 0.0:
        limitations.append("Expected-change counterfactual consistency is zero.")
    if not _control_table_rows(ablations):
        limitations.append("Continuous-scene ablations are not scored.")
    elif qa_v2 is not None and not ablation_provenance_recorded:
        limitations.append(
            "The control aggregate does not record checkpoint/architecture provenance, so "
            "its rows cannot be attributed to v2 solely from their filenames."
        )
    if direct_baseline is None:
        limitations.append("The direct multi-view image baseline is not scored.")
    if oracle_baseline is None:
        limitations.append("The prohibited oracle-text upper bound is not scored.")
    if not robot:
        limitations.append("Robot and MCP benchmarks are not measured.")
    elif not robot.get("semantic_target_navigation_evaluated"):
        limitations.append(
            "The robot benchmark covers numeric mechanics and MCP wiring only; "
            "language-conditioned semantic target navigation is unmeasured."
        )
    limitations.extend(
        (
            (
                "The deterministic robot scan is a pose-dependent numerical map "
                "reobservation, not an arbitrary-pose Blender render plus CLIP remapping."
            ),
            "A center scan reconstructs visible surfaces but cannot reveal occluded rear surfaces.",
        )
    )

    recommendations: list[str] = []
    if inputs.qa_counts.get("validation", 0) == 0 or inputs.qa_counts.get("test", 0) == 0:
        recommendations.append(
            "Generate strictly scene-disjoint non-empty train, validation, and test splits."
        )
    if v1_collapse and qa_v2 is None:
        recommendations.append(
            "Finish v2 anti-collapse training, generate explicitly v2-tagged held-out "
            "predictions, and require changed counterfactual answers before promoting it."
        )
    if qa_v2 is not None:
        recommendations.append(
            "Run v2 wrong-scene, feature-shuffle, position-shuffle, geometry-only, and "
            "empty-prefix controls with checkpoint provenance recorded beside every score."
        )
    elif not v1_collapse:
        recommendations.append(
            "Train and evaluate the multi-scene adapter, then compare its primary result "
            "against scene-content controls."
        )
    if not _control_table_rows(ablations):
        recommendations.append(
            "Score the deterministic geometry, semantic, position, and modality-removal controls."
        )
    if direct_baseline is None or oracle_baseline is None:
        recommendations.append(
            "Run the direct multi-view VLM and isolated oracle-text upper-bound baselines."
        )
    if not robot or not robot.get("semantic_target_navigation_evaluated"):
        recommendations.append(
            "Train and evaluate language-conditioned target-facing and approach behavior "
            "without returning semantic labels through tools."
        )
    recommendations.append(
        "Record peak memory, elapsed time, checkpoint/config provenance, and grounding "
        "calibration for every final experiment."
    )

    lines: list[str] = [
        "# Semantic 3D Chat — First Proof-of-Concept Report",
        "",
        f"Generated from local artifacts on `{generated}`. This report does not run models and does not infer missing measurements.",
        "",
        "## 1. Research question",
        "",
        "Can a local language model answer questions about a synthetic room when the environment reaches it only as continuous, spatially fused visual embeddings and geometry—without a caption, object list, textual scene graph, simulator labels, or question-dependent retrieval?",
        "",
        "The implemented data path satisfies the representation constraint: the runtime consumes continuous scene features and geometry, not environmental text. "
        + evidence_summary,
        "",
        "## 2. Exact architecture",
        "",
        _figure_markdown(figures, "architecture", "Continuous scene-memory architecture"),
        "",
        "The scan is rendered with exact metric camera-Z depth, intrinsics, and camera-to-world poses. Each complete RGB image is encoded once. Middle 768-D, late 768-D, and MaskCLIP-value-aligned 512-D patch streams form a 2,048-D feature. Weighted voxel fusion builds the persistent map before any question. Every occupied block contributes to the question-independent global scene-token set. The selected encoder architecture is `"
        + architecture_version
        + "`, projected directly into the local LM embedding space.",
        "",
        "## 3. Hardware used",
        "",
    ]
    lines += _table(
        ("Item", "Measured value"),
        (
            ("Architecture", machine.get("architecture")),
            ("Processor identifier", machine.get("processor")),
            (
                "Logical / physical CPUs",
                f"{machine.get('logical_cpu_count', 'Not measured')} / {machine.get('physical_cpu_count', 'Not measured')}",
            ),
            ("Unified memory", _gib(machine.get("memory_bytes"))),
            ("Free disk at inspection", _gib(machine.get("disk_free_bytes"))),
            (
                "PyTorch MPS built / available / smoke",
                f"{machine.get('torch_mps_built')} / {machine.get('torch_mps_available')} / {machine.get('torch_mps_smoke')}",
            ),
        ),
    )
    lines += [
        "",
        "The exact Apple chip model is not present in the current machine-report JSON; the report therefore does not guess it.",
        "",
        "## 4. Software versions",
        "",
    ]
    lines += _table(
        ("Component", "Version / revision"),
        (
            ("macOS", machine.get("macos_version", "Not measured")),
            ("Python", machine.get("python_version", "Not measured")),
            ("Blender", machine.get("blender", "Not measured")),
            ("uv", machine.get("uv", "Not measured")),
            ("PyTorch", machine.get("torch_version", "Not measured")),
            (
                "Vision weights",
                f"{models.get('vision', {}).get('model_id', config['vision']['model_id'])} @ {models.get('vision', {}).get('resolved_revision', config['vision'].get('revision', 'Not measured'))}",
            ),
            (
                "Language weights",
                f"{models.get('language', {}).get('model_id', config['language']['model_id'])} @ {models.get('language', {}).get('resolved_revision', config['language'].get('revision', 'Not measured'))}",
            ),
        ),
    )
    lines += [
        "",
        "## 5. Vision encoder selected",
        "",
        f"`{config['vision']['model_id']}` at pinned revision `{config['vision'].get('revision', 'Not measured')}`. One complete {config['vision']['input_size']}×{config['vision']['input_size']} image produces a localized 14×14 patch grid; no manual patch crops are independently encoded. The current aligned slice uses `{config['vision'].get('aligned_method', 'Not measured')}`.",
        "",
        "## 6. Language model selected",
        "",
        f"`{config['language']['model_id']}` at pinned revision `{config['language'].get('revision', 'Not measured')}`. Scene latents are passed through `inputs_embeds`; no scene caption or decoded object list is interposed. CLIP is MIT-licensed and Qwen2.5 is Apache-2.0 according to the project records.",
        "",
        "## 7–11. Representation dimensions and scan scale",
        "",
    ]
    feature_layout = semantic.get("feature_layout", {}) if semantic else {}
    lines += _table(
        ("Parameter", "Value"),
        (
            (
                "Scan images",
                _format_number(inputs.render_frame_count or (map_metrics or {}).get("frame_count")),
            ),
            (
                "Render resolution",
                " × ".join(str(value) for value in config["render"]["resolution"]),
            ),
            ("Feature layout", "middle 768 + late 768 + aligned 512 = 2,048"),
            (
                "Aligned method",
                feature_layout.get(
                    "aligned_method", config["vision"].get("aligned_method", "Not measured")
                ),
            ),
            (
                "Stored semantic dtype",
                (map_metrics or {}).get("semantic_dtype_on_disk", "Not measured"),
            ),
            (
                "Voxel size",
                f"{(map_metrics or {}).get('voxel_size_m', config['mapping']['voxel_size_m']):.3f} m",
            ),
            (
                "Occupied voxels",
                _format_number(
                    (map_metrics or {}).get(
                        "occupied_voxels", semantic.get("voxel_count") if semantic else None
                    )
                ),
            ),
            ("Raw observations", _format_number((map_metrics or {}).get("total_observations"))),
            (
                "Tokenizer input voxels",
                _format_number(
                    startup.get(
                        "processed_voxels", (training or {}).get("tokenizer_input_voxel_count")
                    )
                ),
            ),
            ("Occupied spatial blocks", _format_number(startup.get("occupied_blocks"))),
            (
                "Global scene latents",
                _format_number(
                    checkpoint.get("scene_latents", config["scene_encoder"]["global_latents"])
                ),
            ),
            (
                "Scene encoder dimension",
                _format_number(
                    checkpoint.get("scene_model_dim", config["scene_encoder"]["model_dim"])
                ),
            ),
            ("LM hidden dimension", _format_number(checkpoint.get("language_hidden_dim"))),
            ("Continuous prefix shape", startup.get("prefix_shape", "Not measured")),
        ),
    )
    lines += [
        "",
        _figure_markdown(figures, "scan_montage", "Camera scan montage"),
        "",
        "## 12–13. Training dataset and split",
        "",
        f"QA records: `{_qa_split_summary(inputs)}`. Scene split metadata: `{json.dumps((inputs.qa_splits or {}).get('splits', {}), sort_keys=True)}`.",
        "",
    ]
    if inputs.qa_counts.get("validation", 0) == 0 or inputs.qa_counts.get("test", 0) == 0:
        lines.append(
            "**Critical limitation:** validation and test splits are empty. No held-out scene result can be reported from this artifact set."
        )
    lines += [
        "",
        "## 14. Training",
        "",
    ]
    lines += _table(
        ("Measurement", "Value"),
        (
            ("Selected run namespace", inputs.training_namespace or "root / single-scene"),
            ("Scene-encoder architecture", architecture_version),
            (
                "Completed / target epochs",
                f"{_format_number((training or {}).get('epochs', latest_epoch))} / {_format_number((training or {}).get('target_epochs'))}",
            ),
            ("Latest checkpoint epoch", _format_number(latest_epoch)),
            ("Latest checkpoint train loss", _format_number(latest_loss, 6)),
            ("Best checkpoint epoch", _format_number(checkpoint.get("epoch"))),
            ("Best checkpoint loss", _format_number(checkpoint.get("train_loss"), 6)),
            (
                "Best validation loss",
                _format_number(
                    (training or {}).get("best_validation_loss", checkpoint.get("validation_loss")),
                    6,
                ),
            ),
            (
                "Scenes in checkpoint",
                _format_number(len(checkpoint.get("scene_ids", [])) if checkpoint else None),
            ),
            (
                "Selected-run training time",
                (
                    f"{_format_number((training or {}).get('elapsed_seconds'), 1)} s"
                    if (training or {}).get("elapsed_seconds") is not None
                    else "Not measured"
                ),
            ),
            ("Peak memory", "Not measured"),
        ),
    )
    lines += ["", _figure_markdown(figures, "training_loss", "Training loss curve"), ""]
    if training and latest_epoch and int(training.get("epochs", 0)) != latest_epoch:
        lines += [
            f"The selected training summary records epoch {training.get('epochs')}, while checkpoint metadata reaches epoch {latest_epoch}; the run may still be in progress or the summary may be stale.",
            "",
        ]
    lines += [
        "### Adapter-generation lineage",
        "",
    ]
    if training_v1:
        lines.append(
            f"- **v1 multi-scene:** {training_v1.get('scene_count')} training scenes, "
            f"{training_v1.get('epochs')} epochs, best validation loss "
            f"`{_format_number(training_v1.get('best_validation_loss'), 6)}`, elapsed "
            f"`{_format_number(training_v1.get('elapsed_seconds'), 1)} s`."
        )
    else:
        lines.append("- **v1 multi-scene:** no training summary artifact.")
    if training_v2 or architecture_version == "spatial_coverage_resampler_v2":
        v2_epochs = (training_v2 or training or {}).get("epochs", latest_epoch)
        v2_target = (training_v2 or training or {}).get("target_epochs")
        lines.append(
            f"- **v2 anti-collapse:** architecture `spatial_coverage_resampler_v2`; "
            f"completed/target epochs `{_format_number(v2_epochs)} / {_format_number(v2_target)}`."
        )
    else:
        lines.append("- **v2 anti-collapse:** no training artifact.")
    if v1_collapse:
        lines.append(
            "- The v1 held-out score is retained as a failure result, not promoted as "
            "evidence of scene-conditioned language behavior."
        )
    if resampler_diagnostic:
        factors = [
            pair.get("improvement_factor", {}).get("projected_scene_change")
            for pair in resampler_diagnostic.get("pairs", [])
            if isinstance(pair, Mapping)
        ]
        factors = [float(value) for value in factors if isinstance(value, (int, float))]
        if factors:
            lines.append(
                "- The CPU-only, no-training v2 diagnostic increased projected pairwise "
                f"scene-change magnitude by `{min(factors):.1f}×–{max(factors):.1f}×`. "
                "This is structural evidence only, not a QA result."
            )
    lines.append("")
    lines += [
        "## 15. Static QA results",
        "",
    ]
    if qa:
        lines += _table(
            ("Metric", "Result"),
            (
                ("Normalized exact accuracy", _percent(qa.get("normalized_exact_accuracy"))),
                (
                    "Order-insensitive list accuracy",
                    _percent(qa.get("list_order_insensitive_accuracy")),
                ),
                ("Count accuracy", _percent(qa.get("count", {}).get("accuracy"))),
                ("Spatial-relation accuracy", _percent(qa.get("spatial_relation_accuracy"))),
                ("Presence precision", _percent(qa.get("presence", {}).get("precision"))),
                ("Presence recall", _percent(qa.get("presence", {}).get("recall"))),
            ),
        )
        lines += ["", _figure_markdown(figures, "accuracy_by_type", "Accuracy by question type")]
        lines += [
            "",
            f"Artifact lineage: **{qa_lineage}** (`{qa.get('predictions_path', 'path not recorded')}`).",
        ]
        if qa_lineage == "v1" and v1_collapse:
            lines += [
                "",
                (
                    "**Interpretation:** these are genuine held-out structured scores, but "
                    "they do not demonstrate use of scene-specific content. The v1 wrong-"
                    "scene, semantic-shuffle, position-shuffle, and geometry-only controls "
                    "matched or slightly exceeded the primary score."
                ),
            ]
    else:
        lines.append("**Not measured.** No held-out prediction metrics JSON is present.")
        if inputs.qa_counts.get("test", 0) == 0:
            lines.append("The test split is also empty.")
    lines += [
        "",
        "### Semantic-map prerequisite",
        "",
    ]
    if semantic:
        lines += _table(
            ("Metric", "Observed", "Random control", "Lift"),
            (
                (
                    "Top-1 localization",
                    _percent(semantic_aggregate.get("top1_localization_accuracy")),
                    _percent(semantic_aggregate.get("mean_random_top1_probability")),
                    _percent(semantic_aggregate.get("top1_accuracy_minus_random")),
                ),
                (
                    f"Hit@{semantic.get('top_k', 'k')}",
                    _percent(semantic_aggregate.get("top_k_localization_accuracy")),
                    _percent(semantic_aggregate.get("mean_random_hit_at_k_probability")),
                    _percent(semantic_aggregate.get("top_k_accuracy_minus_random")),
                ),
                (
                    f"Precision@{semantic.get('top_k', 'k')}",
                    _percent(semantic_aggregate.get("mean_precision_at_k")),
                    _percent(semantic_aggregate.get("mean_random_precision_at_k")),
                    _percent(semantic_aggregate.get("precision_at_k_minus_random")),
                ),
            ),
        )
        lines += [
            "",
            f"Cross-view consistency: same-voxel cosine `{_format_number((consistency.get('same_voxel_similarity') or {}).get('mean'))}` versus different-voxel `{_format_number((consistency.get('different_voxel_similarity') or {}).get('mean'))}`; margin `{_format_number(consistency.get('same_minus_different_mean'))}` across `{_format_number(consistency.get('same_voxel_pair_count'))}` same-voxel view pairs.",
            "",
            _figure_markdown(figures, "semantic_by_category", "Semantic localization by category"),
        ]
    else:
        lines.append("Not measured.")
    lines += [
        "",
        "## 16. Counterfactual results",
        "",
    ]
    if counterfactual.get("eligible_pairs"):
        lines += _table(
            ("Metric", "Result"),
            (
                ("Eligible pairs", counterfactual.get("eligible_pairs")),
                ("Pair accuracy", _percent(counterfactual.get("pair_accuracy"))),
                (
                    "Changed when expected",
                    _percent(counterfactual.get("changed_when_expected_rate")),
                ),
                (
                    "Invariant when expected",
                    _percent(counterfactual.get("invariant_when_expected_rate")),
                ),
            ),
        )
        lines += ["", _figure_markdown(figures, "counterfactual", "Counterfactual consistency")]
        if (
            counterfactual.get("expected_change_pairs")
            and float(counterfactual.get("changed_when_expected_rate", 0.0)) == 0.0
        ):
            lines += [
                "",
                (
                    "**Failure:** none of the expected-change pairs changed answer. High "
                    "aggregate pair accuracy is dominated by invariant pairs and must not be "
                    "read as counterfactual success."
                ),
            ]
    else:
        lines.append(
            "**Not measured.** Counterfactual scene-generation controls exist, but no paired QA prediction metrics are present."
        )
    lines += [
        "",
        "## 17. Grounding results",
        "",
    ]
    if qa_grounding.get("target_count"):
        lines += _table(
            ("Metric", "Result"),
            (
                ("Grounding coverage", _percent(qa_grounding.get("coverage"))),
                (
                    "Mean coordinate error",
                    f"{_format_number(qa_grounding.get('mean_coordinate_error_m'))} m",
                ),
                (
                    "Median coordinate error",
                    f"{_format_number(qa_grounding.get('median_coordinate_error_m'))} m",
                ),
                ("Within 0.5 m", _percent(qa_grounding.get("within_0_50m_accuracy"))),
            ),
        )
    else:
        lines.append(
            "**Not measured.** Chat emits numeric grounding coordinates, but no held-out coordinate-error scoring artifact exists."
        )
    lines += [
        "",
        "## 18. Ablation results",
        "",
    ]
    control_rows = _control_table_rows(ablations)
    if ablations and control_rows:
        lines += _table(
            (
                "Condition",
                "Exact",
                "Spatial relation",
                "Changed when expected",
                "Exact Δ vs primary",
            ),
            control_rows,
        )
        lines += ["", _figure_markdown(figures, "ablations", "Ablation accuracy")]
        interpretation = ablations.get("interpretation", {})
        lines += [
            "",
            (
                "Control provenance: checkpoint `"
                + str(ablation_checkpoint)
                + "`, architecture `"
                + str(ablation_architecture)
                + "`."
                if ablation_provenance_recorded
                else (
                    "**Provenance limitation:** this aggregate does not record its checkpoint "
                    "or architecture. The numeric rows are not automatically attributed to "
                    "the selected v2 checkpoint."
                )
            ),
        ]
        if interpretation.get("warning"):
            lines += [
                "",
                f"Artifact-supplied interpretation: **{interpretation['warning']}**",
            ]
    else:
        lines.append(
            "**Not measured.** Deterministic geometry/semantic/XYZ shuffle and zero-semantics/RGB/normals/XYZ map generators exist, but scored outputs are absent."
        )
    if signal_audit:
        findings = signal_audit.get("summary_findings", {})
        lines += [
            "",
            "### v1 collapse diagnosis",
            "",
            str(
                findings.get(
                    "diagnosis",
                    "A scene-signal audit artifact exists, but it contains no textual diagnosis.",
                )
            ),
        ]
        attenuation = findings.get("raw_to_projected_attenuation_factor_range")
        if isinstance(attenuation, list) and len(attenuation) == 2:
            lines.append(
                f"Measured raw-to-projected attenuation: `{attenuation[0]:.1f}×–{attenuation[1]:.1f}×`."
            )
    if resampler_diagnostic:
        lines += [
            "",
            "### v2 structural diagnostic",
            "",
            (
                "The v2 artifact compares the legacy checkpoint under the old and new "
                "resamplers without retraining. It tests signal preservation only; it must "
                "not be cited as held-out language behavior."
            ),
        ]
        diagnostic_rows = []
        for pair in resampler_diagnostic.get("pairs", []):
            if not isinstance(pair, Mapping):
                continue
            improvement = pair.get("improvement_factor", {})
            diagnostic_rows.append(
                (
                    pair.get("change_type", pair.get("pair_id")),
                    _format_number(improvement.get("native_scene_change"), 1),
                    _format_number(improvement.get("projected_scene_change"), 1),
                    _format_number(pair.get("after", {}).get("native_mean_off_diagonal_cosine"), 4),
                    _format_number(
                        pair.get("after", {}).get("projected_mean_off_diagonal_cosine"), 4
                    ),
                )
            )
        if diagnostic_rows:
            lines += _table(
                (
                    "Pair change",
                    "Native signal gain",
                    "Projected signal gain",
                    "v2 native latent cosine",
                    "v2 projected cosine",
                ),
                diagnostic_rows,
            )
    lines += [
        "",
        "## 19. Direct multi-view image baseline",
        "",
        "Not measured." if direct_baseline is None else _json_block(direct_baseline),
        (
            "This evaluation-only control receives complete RGB views directly. It is not "
            "the primary continuous-3D path."
            if direct_baseline is not None
            else ""
        ),
        "",
        "## 20. Oracle-text upper bound",
        "",
        "Not measured." if oracle_baseline is None else _json_block(oracle_baseline),
        (
            "This prohibited evaluation-only upper bound deliberately receives oracle text; "
            "it is isolated from chat inference."
            if oracle_baseline is not None
            else ""
        ),
        "",
        "## 21. Leakage-test results",
        "",
    ]
    if leakage:
        lines += _table(
            ("Control", "Result"),
            (
                ("Overall leakage test", "PASS" if leakage.get("passed") else "FAIL"),
                (
                    "Oracle unavailable during inference",
                    leakage.get("oracle_unavailable_during_inference"),
                ),
                ("Oracle restored", leakage.get("oracle_restored")),
                ("Forbidden accesses", len(leakage.get("forbidden_accesses", []))),
                (
                    "Prefix built before first question",
                    leakage.get("prefix_computed_before_first_question"),
                ),
                ("Prefix invariant", leakage.get("prefix_invariant")),
                ("Prefix hash", leakage.get("prefix_hash")),
                ("Audited loaded files", len(leakage.get("loaded_files", []))),
            ),
        )
    else:
        lines.append("Not measured.")
    lines += [
        "",
        "## 22. Robot-navigation results",
        "",
    ]
    if robot:
        lines += _table(
            ("Measurement", "Result"),
            (
                ("Benchmark scope", robot.get("benchmark_scope", "Not recorded")),
                ("Checks passed", f"{robot.get('passed')} / {robot.get('total')}"),
                ("Pass rate", _percent(robot.get("pass_rate"))),
                ("Trajectory steps", _format_number(robot.get("trajectory_steps"))),
                ("MCP tools registered", _format_number(robot.get("mcp_tool_count"))),
                ("MCP SDK", robot.get("mcp_sdk_version", "Not recorded")),
                (
                    "Semantic target navigation evaluated",
                    robot.get("semantic_target_navigation_evaluated", False),
                ),
            ),
        )
        lines += ["", "Artifact snapshot:", "", _json_block(robot)]
        if not robot.get("semantic_target_navigation_evaluated"):
            lines += [
                "",
                (
                    "This is a bounded numeric action, collision, scan-update, reset, and MCP "
                    "wiring benchmark. It does **not** demonstrate that the chatbot can "
                    "navigate to a named object or follow language-conditioned semantic "
                    "directions."
                ),
                "The measured mechanics do not change the central limitation: language-conditioned semantic target navigation remains unmeasured.",
            ]
    else:
        lines.append("**Not measured.** No robot-navigation metrics artifact is present.")
    lines += [
        "",
        "## 23. Representative conversations",
        "",
    ]
    if inputs.sample_chats:
        lines += _table(
            ("Question", "Answer", "Grounding XYZ (m)", "Prefix hash"),
            [
                (
                    record.get("question", ""),
                    record.get("answer", ""),
                    record.get("grounding_xyz_m", "Not measured"),
                    str(record.get("prefix_hash", ""))[:16] + "…",
                )
                for record in inputs.sample_chats[:8]
            ],
        )
        lines += [
            "",
            (
                "These examples demonstrate runnable local inference only. Their correctness "
                "is not inferred from fluency; structured held-out metrics are reported "
                "separately."
            ),
        ]
    else:
        lines.append("Not recorded.")
    semantic_failures = [
        str(record.get("category"))
        for record in (semantic.get("queries", []) if semantic else [])
        if isinstance(record, Mapping) and not record.get("hit_at_k")
    ]
    lines += [
        "",
        "## 24. Representative failures",
        "",
        (
            f"Semantic localization missed hit@k for: `{', '.join(semantic_failures)}`."
            if semantic_failures
            else "No per-category semantic-localization miss is present in the current artifact."
        ),
        "The initial tokenwise CLIP patch projection failed the semantic sanity gate and was replaced by MaskCLIP-style final-block value features. The obsolete numeric run is not promoted as a current result.",
        (
            "The v1 adapter reached a high raw held-out score but failed scene-content "
            "controls; it learned a near-constant nonzero soft-prompt/prior solution."
            if v1_collapse
            else "No v1 collapse diagnosis is present in the available artifacts."
        ),
        "Fluent chat samples must not be treated as evidence of scene understanding; only the structured held-out and control measurements support behavioral claims.",
        "",
        "## 25. Evidence that the prefix is question-independent",
        "",
        (
            f"PASS for checkpoint `{leakage.get('checkpoint', 'not recorded')}`. Prefix `{leakage.get('prefix_hash')}` was constructed before the first question and remained identical across {leakage.get('question_count')} questions."
            if leakage
            and leakage.get("prefix_invariant")
            and leakage.get("prefix_computed_before_first_question")
            else "Not demonstrated by the available leakage artifact."
        ),
        "",
        "## 26. Evidence that oracle deletion does not break chat",
        "",
        (
            f"PASS for checkpoint `{leakage.get('checkpoint', 'not recorded')}`. The oracle directory was atomically renamed away during local inference, no forbidden path was opened, answers completed, and the directory was restored. This result is not automatically transferred to a different checkpoint without rerunning the test."
            if leakage
            and leakage.get("oracle_unavailable_during_inference")
            and leakage.get("passed")
            else "Not demonstrated by the available leakage artifact."
        ),
        "",
        "## 27. Exact remaining limitations",
        "",
    ]
    lines += [f"- {limitation}" for limitation in limitations]
    if semantic_failures:
        lines.append(
            "- CLIP patch semantics missed the current top-k query set for: "
            + ", ".join(semantic_failures)
            + "."
        )
    lines += ["", "## 28. Recommended next experiments", ""]
    lines += [
        f"{index}. {recommendation}"
        for index, recommendation in enumerate(recommendations, start=1)
    ]
    lines += ["", "## Geometry validation detail", ""]
    if geometry:
        lines += _table(
            ("Metric", "Value"),
            (
                ("Validation status", "PASS" if geometry.get("passed") else "FAIL"),
                ("Sampled points", _format_number(geometry.get("sampled_points"))),
                ("Inside-room fraction", _percent(geometry.get("inside_room_fraction"), 3)),
                (
                    "Reprojection RMSE",
                    f"{_format_number(geometry.get('reprojection_rmse_pixels'), 8)} px",
                ),
                (
                    "Depth round-trip RMSE",
                    f"{_format_number(geometry.get('depth_roundtrip_rmse_m'), 10)} m",
                ),
                (
                    "Cube median surface error",
                    f"{_format_number(geometry.get('cube_surface_median_error_m'), 10)} m",
                ),
            ),
        )
    else:
        lines.append("Not measured.")
    persisted_hashes = (
        _persisted_map_hash_pair(map_metrics, semantic) if map_metrics and semantic else None
    )
    if persisted_hashes and persisted_hashes[0] != persisted_hashes[1]:
        lines += [
            "",
            "### Artifact-version warning",
            "",
            (
                "The persisted numeric-array hash in the mapping summary differs "
                "from the persisted numeric-array hash authenticated by semantic "
                "sanity. These results may refer to different map artifacts; "
                "regenerate the reports from one synchronized map."
            ),
        ]
    if startup.get("warnings"):
        lines += ["", "### Runtime warnings", ""] + [
            f"- {warning}" for warning in startup["warnings"]
        ]
    if inputs.warnings:
        lines += ["", "## Report-builder warnings", ""] + [
            f"- {warning}" for warning in inputs.warnings
        ]
    lines += [
        "",
        "## Artifact inventory",
        "",
        f"Present sources: `{json.dumps(inputs.sources, sort_keys=True)}`",
        "",
        f"Missing metric groups: `{', '.join(inputs.missing) if inputs.missing else 'none'}`",
        "",
    ]
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_report(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    scene_id: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    inputs = collect_report_inputs(project_root, config, scene_id=scene_id)
    figures = generate_report_figures(inputs, config)
    output = (
        Path(output_path).resolve()
        if output_path is not None
        else inputs.reports_root / "final_report.md"
    )
    report = render_final_report(inputs, config, figures)
    _atomic_text(output, report + "\n")
    manifest = {
        "schema_version": 2,
        "report_path": str(output),
        "scene_id": inputs.scene_id,
        "selected_training_namespace": inputs.training_namespace,
        "qa_lineage": _qa_lineage(inputs.metrics.get("qa")),
        "v1_collapse_evidence_present": _v1_collapse_evidence(inputs),
        "sources": inputs.sources,
        "missing_metric_groups": list(inputs.missing),
        "warnings": list(inputs.warnings),
        "figures": figures,
    }
    manifest_path = inputs.reports_root / "metrics" / "report_manifest.json"
    _atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return {**manifest, "manifest_path": str(manifest_path)}
