#!/usr/bin/env python3
"""Plot the sealed V75 official-validation aggregates without re-evaluation.

This script reads exactly one already-sealed score JSON. It does not open model
weights, predictions, references, questions, scene maps, QA files, or oracle data.
The resulting figures are post-hoc visualizations, not a new evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SEALED_SCORE_SHA256 = "f6d9ceea78622c3a4851c3366ac06ed0835824f7b786424725fbfa5d5b978679"
DEFAULT_SCORE = Path("reports/gemma4/metrics/v75_official_validation_score.json")
DEFAULT_OUTPUT_DIR = Path("reports/gemma4/figures")
DEFAULT_MANIFEST = Path("reports/gemma4/metrics/v75_official_validation_figures.json")
FIGURE_FILENAMES = {
    "canonical_accuracy_by_type": "v75_official_validation_accuracy_by_type.png",
    "counterfactual_outcomes": "v75_official_validation_counterfactuals.png",
    "grounding_aggregate_summary": "v75_official_validation_grounding_summary.png",
}
TYPE_ORDER = (
    "attribute",
    "count",
    "metric",
    "orientation",
    "presence",
    "spatial_relation",
    "support",
)
FAMILY_ORDER = ("book_support", "mirror_lr", "picture_support")
PNG_METADATA = {"Software": "semantic_3d_chat deterministic post-hoc plotter"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_fraction(record: Mapping[str, Any], *, name: str) -> tuple[int, int, float]:
    correct = record.get("correct")
    total = record.get("total")
    accuracy = _finite_float(record.get("accuracy"), name=f"{name} accuracy")
    if (
        isinstance(correct, bool)
        or not isinstance(correct, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or not 0 <= correct <= total
        or not math.isclose(accuracy, correct / total, abs_tol=1e-12)
    ):
        raise ValueError(f"{name} count/accuracy contract differs")
    return correct, total, accuracy


def load_sealed_score(path: Path) -> dict[str, Any]:
    """Read and validate the one authorized aggregate score artifact."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Sealed score must be one regular file: {path}")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != SEALED_SCORE_SHA256:
        raise ValueError(
            "V75 official-validation score digest differs: "
            f"expected {SEALED_SCORE_SHA256}, observed {observed_sha256}"
        )
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError("V75 official-validation score must contain one JSON object")
    metrics = payload.get("metrics")
    scope = payload.get("scope")
    if not isinstance(metrics, Mapping) or not isinstance(scope, Mapping):
        raise TypeError("V75 score metrics/scope must be mappings")
    canonical = metrics.get("canonical")
    changed = metrics.get("changed_counterfactual")
    standard = metrics.get("standard")
    if not all(isinstance(value, Mapping) for value in (canonical, changed, standard)):
        raise TypeError("V75 score aggregate sections must be mappings")
    per_type = canonical.get("per_type")
    by_family = changed.get("by_family")
    grounding = standard.get("grounding")
    if not all(isinstance(value, Mapping) for value in (per_type, by_family, grounding)):
        raise TypeError("V75 score plot inputs must be mappings")
    if not (
        payload.get("schema_version") == 1
        and payload.get("artifact") == "v75_official_validation_score_v1"
        and payload.get("passed") is False
        and scope.get("split") == "validation"
        and scope.get("candidate_count") == 1
        and scope.get("question_count") == 216
        and scope.get("model_loaded") is False
        and scope.get("scene_map_loaded") is False
        and scope.get("simulator_oracle_loaded") is False
        and scope.get("question_or_answer_text_serialized") is False
        and tuple(per_type) == TYPE_ORDER
        and tuple(by_family) == FAMILY_ORDER
    ):
        raise ValueError("V75 official-validation score identity or scope differs")
    total_correct = 0
    total_questions = 0
    for answer_type in TYPE_ORDER:
        record = per_type[answer_type]
        if not isinstance(record, Mapping):
            raise TypeError(f"V75 per-type record is not a mapping: {answer_type}")
        correct, total, _accuracy = _exact_fraction(record, name=answer_type)
        total_correct += correct
        total_questions += total
    overall_correct, overall_total, overall_accuracy = _exact_fraction(
        canonical, name="canonical overall"
    )
    if not (
        (overall_correct, overall_total) == (167, 216)
        and (total_correct, total_questions) == (167, 216)
        and math.isclose(overall_accuracy, 167 / 216, abs_tol=1e-12)
    ):
        raise ValueError("V75 canonical aggregate differs")
    return payload


def _new_figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), dpi=160, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(
        temporary,
        format="png",
        dpi=160,
        facecolor="white",
        metadata=PNG_METADATA,
    )
    os.replace(temporary, path)
    figure.clear()


def plot_canonical_accuracy(score: Mapping[str, Any], path: Path) -> None:
    per_type = score["metrics"]["canonical"]["per_type"]
    labels = [name.replace("_", " ").title() for name in TYPE_ORDER]
    rows = [per_type[name] for name in TYPE_ORDER]
    values = [float(row["accuracy"]) for row in rows]
    colors = ["#dc2626" if name == "spatial_relation" else "#2563eb" for name in TYPE_ORDER]

    figure = _new_figure(10.8, 6.5)
    axis = figure.add_subplot(1, 1, 1)
    bars = axis.bar(labels, values, color=colors, edgecolor="#0f172a", linewidth=0.6)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Canonical accuracy")
    axis.set_title(
        "V75 official validation · canonical accuracy by question type\n"
        "Post-hoc visualization of sealed score — not a new evaluation",
        fontsize=13,
        pad=14,
    )
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.tick_params(axis="x", labelrotation=24)
    for bar, row in zip(bars, rows, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.018,
            f"{row['correct']}/{row['total']}\n{100.0 * row['accuracy']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    spatial_index = TYPE_ORDER.index("spatial_relation")
    axis.scatter(
        [spatial_index],
        [score["thresholds"]["spatial_relation_accuracy_minimum"]],
        marker="_",
        s=650,
        linewidths=2.4,
        color="#111827",
        zorder=5,
        label="Locked spatial minimum (60%)",
    )
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.21)
    _save_figure(figure, path)


def plot_counterfactual_outcomes(score: Mapping[str, Any], path: Path) -> None:
    changed = score["metrics"]["changed_counterfactual"]
    families = changed["by_family"]
    labels = [name.replace("_", " ").title() for name in FAMILY_ORDER]
    measures = (
        ("Complete paired units", "complete_units", 1, "#2563eb"),
        ("Correct sides", "correct_sides", 2, "#16a34a"),
        ("Prediction changed", "prediction_changed_units", 1, "#d97706"),
    )
    x_positions = list(range(len(FAMILY_ORDER)))
    width = 0.24

    figure = _new_figure(10.4, 6.4)
    axis = figure.add_subplot(1, 1, 1)
    for measure_index, (label, field, denominator_scale, color) in enumerate(measures):
        offsets = [x + (measure_index - 1) * width for x in x_positions]
        numerators = [int(families[name][field]) for name in FAMILY_ORDER]
        denominators = [
            int(families[name]["unit_count"]) * denominator_scale for name in FAMILY_ORDER
        ]
        values = [
            numerator / denominator
            for numerator, denominator in zip(numerators, denominators, strict=True)
        ]
        bars = axis.bar(
            offsets,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="#0f172a",
            linewidth=0.5,
        )
        for bar, numerator, denominator in zip(bars, numerators, denominators, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{numerator}/{denominator}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    axis.set_xticks(x_positions, labels)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Outcome rate")
    axis.set_title(
        "V75 official validation · changed counterfactual outcomes\n"
        "Post-hoc visualization of sealed score — 12 units / 24 sides",
        fontsize=13,
        pad=14,
    )
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.14)
    _save_figure(figure, path)


def plot_grounding_summary(score: Mapping[str, Any], path: Path) -> None:
    grounding = score["metrics"]["standard"]["grounding"]
    errors = (
        float(grounding["mean_coordinate_error_m"]),
        float(grounding["median_coordinate_error_m"]),
        float(grounding["rmse_coordinate_error_m"]),
    )
    threshold_rates = (
        float(grounding["within_0_25m_accuracy"]),
        float(grounding["within_0_50m_accuracy"]),
        float(grounding["within_1_00m_accuracy"]),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in (*errors, *threshold_rates)):
        raise ValueError("V75 grounding aggregates must be finite and nonnegative")

    figure = _new_figure(11.0, 5.9)
    left = figure.add_subplot(1, 2, 1)
    right = figure.add_subplot(1, 2, 2)
    error_bars = left.bar(
        ["Mean", "Median", "RMSE"],
        errors,
        color=["#2563eb", "#7c3aed", "#db2777"],
        edgecolor="#0f172a",
        linewidth=0.6,
    )
    left.set_ylabel("Coordinate error (m)")
    left.set_ylim(0.0, max(errors) * 1.22)
    left.grid(axis="y", alpha=0.25, linewidth=0.7)
    for bar, value in zip(error_bars, errors, strict=True):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(errors) * 0.035,
            f"{value:.3f} m",
            ha="center",
            fontsize=9,
        )

    hit_bars = right.bar(
        ["≤0.25 m", "≤0.50 m", "≤1.00 m"],
        threshold_rates,
        color="#64748b",
        edgecolor="#0f172a",
        linewidth=0.6,
    )
    right.set_ylabel("Fraction of 132 grounded targets")
    right.set_ylim(0.0, 1.0)
    right.grid(axis="y", alpha=0.25, linewidth=0.7)
    for bar, value in zip(hit_bars, threshold_rates, strict=True):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            max(value + 0.025, 0.025),
            f"{100.0 * value:.1f}%",
            ha="center",
            fontsize=9,
        )
    figure.suptitle(
        "V75 official validation · aggregate grounding summary\n"
        "No per-example errors are serialized; this is not a distribution or new evaluation",
        fontsize=13,
        y=0.96,
    )
    figure.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.14, wspace=0.28)
    _save_figure(figure, path)


def generate_figures(
    score_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    score = load_sealed_score(score_path)
    outputs = {name: output_dir / filename for name, filename in FIGURE_FILENAMES.items()}
    plot_canonical_accuracy(score, outputs["canonical_accuracy_by_type"])
    plot_counterfactual_outcomes(score, outputs["counterfactual_outcomes"])
    plot_grounding_summary(score, outputs["grounding_aggregate_summary"])

    captions = {
        "canonical_accuracy_by_type": (
            "Canonical accuracy by question type from the sealed V75 official-validation score."
        ),
        "counterfactual_outcomes": (
            "Per-family complete-unit, correct-side, and prediction-change rates for "
            "the sealed official counterfactual subset."
        ),
        "grounding_aggregate_summary": (
            "Aggregate grounding errors and threshold hit rates. The sealed score "
            "contains no per-example errors, so no distribution is inferred."
        ),
    }
    manifest = {
        "artifact": "v75_official_validation_posthoc_figures_v1",
        "schema_version": 1,
        "source": {
            "path": score_path.as_posix(),
            "sha256": SEALED_SCORE_SHA256,
            "artifact": score["artifact"],
        },
        "scope": {
            "post_hoc_visualization_only": True,
            "new_evaluation": False,
            "source_file_count": 1,
            "model_loaded": False,
            "predictions_or_references_loaded": False,
            "scene_map_loaded": False,
            "qa_or_oracle_loaded": False,
            "unopened_split_loaded": False,
            "per_example_grounding_errors_available": False,
            "grounding_visualization": "aggregate_summary_only_no_distribution",
        },
        "figures": {
            name: {
                "path": path.as_posix(),
                "sha256": file_sha256(path),
                "caption": captions[name],
            }
            for name, path in outputs.items()
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    manifest = generate_figures(args.score, args.output_dir, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
