#!/usr/bin/env python3
"""Plot sealed V78 historical-held grounding/control aggregates.

This is a deterministic post-hoc plot of one internal historical-held report.
It is not official validation, a runtime result, or a promotion decision.
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

SEALED_REPORT_SHA256 = "557cc497dd12bd74f45cecd3624e18649ddc548af091ed719c22c7998942b84b"
DEFAULT_REPORT = Path("reports/gemma4/metrics/v78_grounding_sidecar_internal_held.json")
DEFAULT_FIGURE = Path("reports/gemma4/figures/v78_grounding_internal_held_controls.png")
DEFAULT_MANIFEST = Path("reports/gemma4/metrics/v78_grounding_internal_held_figure.json")
CONDITIONS = (
    ("historical_internal_held", "V78 candidate", "#2563eb"),
    ("paired_wrong_scene", "Wrong paired scene", "#d97706"),
    ("v54_same_historical_internal_held", "V54 baseline", "#64748b"),
    ("zero_scene_same_historical_internal_held", "Zero scene", "#94a3b8"),
    ("scene_token_position_shuffle", "Position shuffle", "#dc2626"),
    ("question_embedding_shuffle", "Question shuffle", "#7c3aed"),
)
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


def load_sealed_report(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Sealed V78 report must be one regular file: {path}")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != SEALED_REPORT_SHA256:
        raise ValueError(
            "V78 historical-held report digest differs: "
            f"expected {SEALED_REPORT_SHA256}, observed {observed_sha256}"
        )
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError("V78 report must be a mapping")
    metrics = payload.get("metrics")
    split = payload.get("split")
    if not isinstance(metrics, Mapping):
        raise TypeError("V78 metrics must be a mapping")
    if not isinstance(split, Mapping):
        raise TypeError("V78 split must be a mapping")
    if not (
        payload.get("artifact") == "v78_historical_training_grounding_repair_report_v1"
        and payload.get("schema_version") == 1
        and payload.get("status") == "internal_historical_diagnostic_only"
        and payload.get("runtime_promotion_authorized") is False
        and payload.get("official_validation_loaded") is False
        and payload.get("official_test_loaded") is False
        and payload.get("deferred_final_loaded") is False
        and payload.get("oracle_files_loaded") is False
        and payload.get("environmental_text_inputs") == []
        and split.get("held_grounded_rows") == 94
        and split.get("pair_disjoint") is True
        and split.get("scene_disjoint") is True
    ):
        raise ValueError("V78 historical-held identity, isolation, or split differs")
    for key, _label, _color in CONDITIONS:
        record = metrics.get(key)
        if not isinstance(record, Mapping) or record.get("count") != 94:
            raise ValueError(f"V78 control count differs: {key}")
        for field in (
            "mean_coordinate_error_m",
            "within_0_50m_accuracy",
            "within_1m_accuracy",
        ):
            value = _finite_float(record.get(field), name=f"{key} {field}")
            if value < 0.0 or (field.endswith("accuracy") and value > 1.0):
                raise ValueError(f"V78 control metric is out of range: {key} {field}")
    return payload


def plot_report(report: Mapping[str, Any], path: Path) -> None:
    metrics = report["metrics"]
    labels = [label for _key, label, _color in CONDITIONS]
    colors = [color for _key, _label, color in CONDITIONS]
    mean_errors = [
        float(metrics[key]["mean_coordinate_error_m"]) for key, _label, _color in CONDITIONS
    ]
    within_one = [float(metrics[key]["within_1m_accuracy"]) for key, _label, _color in CONDITIONS]

    figure = Figure(figsize=(12.2, 6.3), dpi=160, facecolor="white")
    FigureCanvasAgg(figure)
    left = figure.add_subplot(1, 2, 1)
    right = figure.add_subplot(1, 2, 2)
    x_positions = list(range(len(CONDITIONS)))

    error_bars = left.bar(
        x_positions,
        mean_errors,
        color=colors,
        edgecolor="#0f172a",
        linewidth=0.5,
    )
    left.set_xticks(x_positions, labels, rotation=32, ha="right")
    left.set_ylabel("Mean coordinate error (m) · lower is better")
    left.set_ylim(0.0, max(mean_errors) * 1.18)
    left.grid(axis="y", alpha=0.25, linewidth=0.7)
    for bar, value in zip(error_bars, mean_errors, strict=True):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(mean_errors) * 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )

    hit_bars = right.bar(
        x_positions,
        within_one,
        color=colors,
        edgecolor="#0f172a",
        linewidth=0.5,
    )
    right.set_xticks(x_positions, labels, rotation=32, ha="right")
    right.set_ylabel("Within 1 m accuracy · higher is better")
    right.set_ylim(0.0, 1.08)
    right.grid(axis="y", alpha=0.25, linewidth=0.7)
    for bar, value in zip(hit_bars, within_one, strict=True):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{100.0 * value:.1f}%",
            ha="center",
            fontsize=8,
        )

    figure.suptitle(
        "V78 grounding repair and controls · 94 historical-held rows\n"
        "Internal diagnostic only — not official validation, runtime, or promotion",
        fontsize=13,
        y=0.965,
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.30, wspace=0.28)
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
    manifest_path: Path,
) -> dict[str, Any]:
    report = load_sealed_report(report_path)
    plot_report(report, figure_path)
    manifest = {
        "artifact": "v78_historical_held_grounding_posthoc_figure_v1",
        "schema_version": 1,
        "source": {
            "path": report_path.as_posix(),
            "sha256": SEALED_REPORT_SHA256,
            "artifact": report["artifact"],
        },
        "scope": {
            "historical_internal_held_only": True,
            "post_hoc_visualization_only": True,
            "new_evaluation": False,
            "official_validation": False,
            "runtime_evidence": False,
            "promotion_evidence": False,
            "runtime_promotion_authorized": False,
            "source_file_count": 1,
            "model_loaded": False,
            "predictions_or_references_loaded": False,
            "qa_or_oracle_loaded": False,
            "unopened_split_loaded": False,
        },
        "figure": {
            "path": figure_path.as_posix(),
            "sha256": file_sha256(figure_path),
            "caption": (
                "V78 historical-held grounding candidate versus matched controls. "
                "The paired-wrong-scene aggregate is nearly unchanged (0.534 m "
                "mean error and 92.6% within 1 m) because many rows do not move "
                "their target; only the 10 changed-target sides show 90% "
                "correct-scene preference. Position/question shuffles and zero "
                "scene are the stronger controls. This is internal diagnostic "
                "evidence only."
            ),
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
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    manifest = generate_figure(args.report, args.figure, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
