"""Plot the sealed V85 development aggregate without rerunning evaluation.

This module reads exactly one hash-pinned aggregate JSON.  It does not load the
model, predictions, questions, scene memories, maps, QA, or oracle files.  The
output is a post-hoc visualization of development evidence, not official
validation and not evidence for runtime promotion.
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

SEALED_DEVELOPMENT_SHA256 = (
    "202134d8900e105d63f23d1cc1d19d68a882c4464382b7a63b7aa007f2714828"
)
DEFAULT_DEVELOPMENT_REPORT = Path(
    "reports/gemma4/metrics/gemma4_v85_strict_multiscene_development.json"
)
DEFAULT_FIGURE = Path("reports/gemma4/figures/v85_development_accuracy_by_type.png")
DEFAULT_SUMMARY = Path("reports/gemma4/examples/v85_development_accuracy_by_type.json")
ANSWER_TYPE_ORDER = (
    "attribute",
    "count",
    "metric",
    "orientation",
    "presence",
    "spatial_relation",
    "support",
)
EXPECTED_COUNTS = {
    "attribute": (20, 80),
    "count": (62, 64),
    "metric": (12, 16),
    "orientation": (13, 14),
    "presence": (31, 70),
    "spatial_relation": (42, 80),
    "support": (34, 60),
}
PNG_METADATA = {"Software": "semantic_3d_chat deterministic V85 development plotter"}


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


def load_sealed_development_report(path: Path) -> dict[str, Any]:
    """Authenticate and parse the one authorized V85 aggregate report."""

    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Sealed V85 development report must be one file: {path}")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != SEALED_DEVELOPMENT_SHA256:
        raise ValueError(
            "V85 development report digest differs: "
            f"expected {SEALED_DEVELOPMENT_SHA256}, observed {observed_sha256}"
        )
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError("V85 development report must contain one JSON object")
    metrics = payload.get("metrics")
    split = payload.get("split_preflight")
    leakage = payload.get("leakage")
    scene_memory = payload.get("scene_memory")
    if not all(isinstance(value, Mapping) for value in (metrics, split, leakage, scene_memory)):
        raise TypeError("V85 aggregate sections must be mappings")
    by_type = metrics.get("canonical_accuracy_by_answer_type")
    overall = metrics.get("canonical_type_specific")
    majority = metrics.get("answer_frequency_majority_baseline")
    gates = metrics.get("runtime_candidate_gates")
    if not all(isinstance(value, Mapping) for value in (by_type, overall, majority, gates)):
        raise TypeError("V85 metric sections must be mappings")
    if not (
        payload.get("artifact") == "gemma4_v85_strict_multiscene_development_score_v1"
        and payload.get("schema_version") == 85
        and payload.get("status") == "runtime_candidate_gate_pass_separate_packaging_required"
        and payload.get("official_validation_loaded") is False
        and payload.get("official_test_loaded") is False
        and payload.get("deferred_final_loaded") is False
        and payload.get("oracle_loaded") is False
        and payload.get("runtime_promotion_authorized") is False
        and payload.get("automatic_runtime_promotion") is False
        and payload.get("checkpoint_selection_after_scoring") is False
        and payload.get("fixed_checkpoint_selected_before_development") is True
        and metrics.get("runtime_candidate_gate_passed") is True
        and metrics.get("separate_leakage_runtime_packaging_authorized") is True
        and tuple(by_type) == ANSWER_TYPE_ORDER
        and split.get("development_rows") == 384
        and len(split.get("development_scenes", [])) == 16
        and split.get("pair_and_scene_disjoint") is True
        and leakage.get("protected_read_count") == 0
        and leakage.get("oracle_loaded") is False
        and scene_memory.get("prefix_hash_invariant") is True
        and scene_memory.get("same_prefix_reused_for_every_question") is True
        and scene_memory.get("question_derived_environmental_tokens") == 0
    ):
        raise ValueError("V85 development identity, scope, isolation, or gate differs")

    summed_correct = 0
    summed_total = 0
    for answer_type in ANSWER_TYPE_ORDER:
        row = by_type[answer_type]
        if not isinstance(row, Mapping):
            raise TypeError(f"V85 answer-type row is not a mapping: {answer_type}")
        correct, total, _accuracy = _exact_fraction(row, name=answer_type)
        if (correct, total) != EXPECTED_COUNTS[answer_type]:
            raise ValueError(f"V85 answer-type count differs: {answer_type}")
        summed_correct += correct
        summed_total += total
    overall_correct, overall_total, _overall_accuracy = _exact_fraction(
        overall, name="V85 development overall"
    )
    majority_correct, majority_total, _majority_accuracy = _exact_fraction(
        majority, name="V85 development majority baseline"
    )
    if not (
        (summed_correct, summed_total) == (214, 384)
        and (overall_correct, overall_total) == (214, 384)
        and (majority_correct, majority_total) == (62, 384)
        and gates.get("canonical_accuracy_at_least_preregistered_threshold") is True
        and gates.get("spatial_relation_accuracy_at_least_0_45") is True
    ):
        raise ValueError("V85 development aggregate or preregistered gate differs")
    return payload


def plot_accuracy_by_type(report: Mapping[str, Any], path: Path) -> None:
    metrics = report["metrics"]
    by_type = metrics["canonical_accuracy_by_answer_type"]
    overall = metrics["canonical_type_specific"]
    majority = metrics["answer_frequency_majority_baseline"]
    labels = [name.replace("_", " ").title() for name in ANSWER_TYPE_ORDER]
    rows = [by_type[name] for name in ANSWER_TYPE_ORDER]
    values = [float(row["accuracy"]) for row in rows]
    colors = ["#dc2626" if name == "attribute" else "#2563eb" for name in ANSWER_TYPE_ORDER]

    figure = Figure(figsize=(10.0, 6.2), dpi=160, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    positions = list(range(len(labels)))
    bars = axis.barh(
        positions,
        values,
        color=colors,
        edgecolor="#0f172a",
        linewidth=0.55,
        height=0.68,
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.04)
    axis.set_xlabel("Canonical exact accuracy")
    axis.grid(axis="x", alpha=0.23, linewidth=0.7)
    overall_accuracy = float(overall["accuracy"])
    majority_accuracy = float(majority["accuracy"])
    axis.axvline(
        majority_accuracy,
        color="#64748b",
        linestyle="--",
        linewidth=1.7,
        label=f"Answer-frequency majority · {100.0 * majority_accuracy:.2f}%",
    )
    axis.axvline(
        overall_accuracy,
        color="#0f172a",
        linestyle="-",
        linewidth=1.7,
        label=f"Overall · {100.0 * overall_accuracy:.2f}%",
    )
    for bar, row in zip(bars, rows, strict=True):
        inside = bar.get_width() >= 0.82
        axis.text(
            bar.get_width() - 0.012 if inside else bar.get_width() + 0.014,
            bar.get_y() + bar.get_height() / 2,
            f"{row['correct']}/{row['total']}  ·  {100.0 * row['accuracy']:.1f}%",
            va="center",
            ha="right" if inside else "left",
            fontsize=8.6,
            color="white" if inside else "#111827",
            fontweight="bold" if inside else "normal",
            zorder=6,
            bbox=(
                None
                if inside
                else {
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                    "pad": 1.2,
                }
            ),
        )
    axis.set_title(
        "V85 scene-disjoint development accuracy by answer type\n"
        "Development only · not official validation · candidate not promoted",
        fontsize=13,
        pad=14,
    )
    axis.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, -0.13),
        frameon=False,
        fontsize=8.8,
        ncol=2,
        borderaxespad=0.0,
    )
    figure.text(
        0.99,
        0.012,
        "Post-hoc plot of one sealed aggregate; no new inference",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#475569",
    )
    figure.subplots_adjust(left=0.20, right=0.975, top=0.82, bottom=0.22)
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


def generate_figure(
    report_path: Path,
    figure_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    report = load_sealed_development_report(report_path)
    plot_accuracy_by_type(report, figure_path)
    metrics = report["metrics"]
    by_type = metrics["canonical_accuracy_by_answer_type"]
    overall = metrics["canonical_type_specific"]
    majority = metrics["answer_frequency_majority_baseline"]
    summary = {
        "artifact": "v85_development_accuracy_by_type_posthoc_figure_v1",
        "schema_version": 1,
        "source": {
            "path": report_path.as_posix(),
            "sha256": SEALED_DEVELOPMENT_SHA256,
            "artifact": report["artifact"],
        },
        "scope": {
            "development_only": True,
            "pair_and_scene_disjoint": True,
            "development_scene_count": 16,
            "development_question_count": 384,
            "official_validation": False,
            "official_test": False,
            "deferred_final": False,
            "post_hoc_visualization_only": True,
            "new_evaluation": False,
            "new_inference": False,
            "model_loaded": False,
            "predictions_or_references_loaded": False,
            "scene_memory_or_map_loaded": False,
            "qa_or_oracle_loaded": False,
            "runtime_promotion_authorized": False,
        },
        "metrics": {
            "overall": dict(overall),
            "answer_frequency_majority_baseline": dict(majority),
            "canonical_accuracy_by_answer_type": {
                name: dict(by_type[name]) for name in ANSWER_TYPE_ORDER
            },
        },
        "figure": {
            "path": figure_path.as_posix(),
            "sha256": file_sha256(figure_path),
            "caption": (
                "V85 canonical exact development accuracy by answer type. Overall "
                "accuracy is 214/384 (55.73%) versus the answer-frequency majority "
                "baseline of 62/384 (16.15%). This is scene-disjoint development "
                "evidence only, not official validation and not runtime promotion."
            ),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_DEVELOPMENT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    summary = generate_figure(args.report, args.figure, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
