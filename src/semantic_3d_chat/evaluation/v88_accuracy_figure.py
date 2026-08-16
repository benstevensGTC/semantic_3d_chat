"""Plot the sealed V88 development-known single-scene result without inference."""

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

SEALED_EVALUATION_SHA256 = "40b4c591d84a3b1ae99c301d017e9c6212308203a319e3c511f22583e5a78641"
DEFAULT_EVALUATION_REPORT = Path(
    "reports/gemma4/metrics/gemma4_v88_scene1_augmented_evaluation.json"
)
DEFAULT_FIGURE = Path("reports/gemma4/figures/v88_scene1_accuracy_by_type.png")
DEFAULT_SUMMARY = Path("reports/gemma4/examples/v88_scene1_accuracy_by_type.json")
ANSWER_TYPE_ORDER = (
    "attribute",
    "count",
    "metric",
    "presence",
    "spatial_relation",
    "support",
)
EXPECTED_COUNTS = {
    "attribute": (11, 18),
    "count": (9, 9),
    "metric": (1, 1),
    "presence": (21, 22),
    "spatial_relation": (64, 86),
    "support": (1, 2),
}
PNG_METADATA = {"Software": "semantic_3d_chat deterministic V88 result plotter"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction(record: Mapping[str, Any], *, name: str) -> tuple[int, int, float]:
    correct, total, accuracy = (
        record.get("correct"),
        record.get("total"),
        record.get("accuracy"),
    )
    if (
        isinstance(correct, bool)
        or not isinstance(correct, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or not math.isclose(float(accuracy), correct / total, abs_tol=1e-12)
    ):
        raise ValueError(f"V88 {name} fraction differs")
    return correct, total, float(accuracy)


def load_sealed_evaluation(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Sealed V88 evaluation must be one file: {path}")
    observed = file_sha256(path)
    if observed != SEALED_EVALUATION_SHA256:
        raise ValueError(
            "V88 evaluation digest differs: "
            f"expected {SEALED_EVALUATION_SHA256}, observed {observed}"
        )
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError("V88 evaluation must contain one JSON object")
    metrics = payload.get("metrics")
    memory = payload.get("scene_memory")
    leakage = payload.get("leakage")
    if not all(isinstance(value, Mapping) for value in (metrics, memory, leakage)):
        raise TypeError("V88 result sections must be mappings")
    overall = metrics.get("canonical_type_specific")
    by_type = metrics.get("canonical_accuracy_by_answer_type")
    smoke = metrics.get("generic_smoke")
    causal = metrics.get("causal_control")
    gates = metrics.get("model_acceptance_gates")
    if not all(
        isinstance(value, Mapping)
        for value in (overall, by_type, smoke, causal, gates)
    ):
        raise TypeError("V88 metric sections must be mappings")
    failed = {name for name, passed in gates.items() if passed is False}
    if not (
        payload.get("artifact") == "gemma4_v88_scene1_augmented_evaluation_v1"
        and payload.get("schema_version") == 88
        and payload.get("status") == "model_gates_fail_not_runtime_promotable"
        and payload.get("development_known_smoke_trained") is True
        and payload.get("held_out_smoke_claim") is False
        and payload.get("held_out_generalization_claim") is False
        and payload.get("official_validation_loaded") is False
        and payload.get("official_test_loaded") is False
        and payload.get("deferred_final_loaded") is False
        and payload.get("oracle_loaded") is False
        and payload.get("runtime_promotion_authorized") is False
        and payload.get("separate_runtime_packaging_authorized") is False
        and metrics.get("model_acceptance_gate_passed") is False
        and failed == {"all_scene1_canonical_accuracy_at_least_0_80"}
        and smoke.get("correct") == 3
        and smoke.get("total") == 3
        and smoke.get("development_known_and_trained") is True
        and smoke.get("held_out") is False
        and causal.get("canonical_prediction_changes") == 2
        and math.isclose(
            float(causal.get("mean_zero_minus_correct_nll")),
            1.8954026500384014,
            abs_tol=1e-12,
        )
        and tuple(by_type) == ANSWER_TYPE_ORDER
        and memory.get("prefix_hash_invariant") is True
        and memory.get("same_prefix_reused_for_every_question") is True
        and memory.get("question_derived_environmental_tokens") == 0
        and leakage.get("protected_read_count") == 0
        and leakage.get("oracle_loaded") is False
    ):
        raise ValueError("V88 identity, scope, causal, isolation, or gate differs")
    summed = [0, 0]
    for answer_type in ANSWER_TYPE_ORDER:
        row = by_type[answer_type]
        if not isinstance(row, Mapping):
            raise TypeError(f"V88 answer-type row is not a mapping: {answer_type}")
        correct, total, _accuracy = _fraction(row, name=answer_type)
        if (correct, total) != EXPECTED_COUNTS[answer_type]:
            raise ValueError(f"V88 answer-type count differs: {answer_type}")
        summed[0] += correct
        summed[1] += total
    if tuple(summed) != (107, 138) or _fraction(overall, name="overall")[:2] != (107, 138):
        raise ValueError("V88 aggregate differs")
    return payload


def plot_accuracy_by_type(report: Mapping[str, Any], path: Path) -> None:
    metrics = report["metrics"]
    by_type = metrics["canonical_accuracy_by_answer_type"]
    overall = metrics["canonical_type_specific"]
    rows = [by_type[name] for name in ANSWER_TYPE_ORDER]
    labels = [name.replace("_", " ").title() for name in ANSWER_TYPE_ORDER]
    values = [float(row["accuracy"]) for row in rows]
    colors = ["#16a34a" if value >= 0.8 else "#dc2626" for value in values]

    figure = Figure(figsize=(10.0, 5.8), dpi=160, facecolor="white")
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
    axis.axvline(
        0.8,
        color="#475569",
        linestyle="--",
        linewidth=1.7,
        label="Locked overall threshold · 80.00%",
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
            fontsize=8.7,
            color="white" if inside else "#111827",
            fontweight="bold" if inside else "normal",
            zorder=6,
            bbox=(
                None
                if inside
                else {"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.2}
            ),
        )
    axis.set_title(
        "V88 augmented single-scene accuracy by answer type\n"
        "Development-known training set · overall gate failed · not promoted",
        fontsize=13,
        pad=14,
    )
    axis.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, -0.14),
        frameon=False,
        fontsize=8.8,
        ncol=2,
        borderaxespad=0.0,
    )
    figure.text(
        0.99,
        0.012,
        "Post-hoc sealed aggregate · trained smoke · no inference or held-out claim",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#475569",
    )
    figure.subplots_adjust(left=0.20, right=0.975, top=0.80, bottom=0.23)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    figure.savefig(temporary, format="png", dpi=160, facecolor="white", metadata=PNG_METADATA)
    os.replace(temporary, path)
    figure.clear()


def generate_figure(report_path: Path, figure_path: Path, summary_path: Path) -> dict[str, Any]:
    report = load_sealed_evaluation(report_path)
    plot_accuracy_by_type(report, figure_path)
    metrics = report["metrics"]
    summary = {
        "artifact": "v88_scene1_accuracy_by_type_posthoc_figure_v1",
        "schema_version": 1,
        "source": {
            "path": report_path.as_posix(),
            "sha256": SEALED_EVALUATION_SHA256,
            "artifact": report["artifact"],
        },
        "scope": {
            "single_scene_training_authorized_evaluation": True,
            "development_known_smoke_trained": True,
            "held_out_smoke": False,
            "held_out_generalization": False,
            "official_validation": False,
            "official_test": False,
            "post_hoc_visualization_only": True,
            "new_evaluation": False,
            "new_inference": False,
            "model_loaded": False,
            "predictions_or_references_loaded": False,
            "qa_or_oracle_loaded": False,
            "runtime_promotion_authorized": False,
        },
        "metrics": {
            "overall": dict(metrics["canonical_type_specific"]),
            "overall_acceptance_threshold": 0.8,
            "acceptance_passed": False,
            "failed_model_gates": ["all_scene1_canonical_accuracy_at_least_0_80"],
            "canonical_accuracy_by_answer_type": {
                name: dict(metrics["canonical_accuracy_by_answer_type"][name])
                for name in ANSWER_TYPE_ORDER
            },
        },
        "figure": {
            "path": figure_path.as_posix(),
            "sha256": file_sha256(figure_path),
            "caption": (
                "V88 development-known single-scene accuracy by answer type. Overall "
                "accuracy was 107/138 (77.54%), below the locked 80% gate. The 3/3 "
                "smoke was trained and is not held out; this is not generalization, "
                "official validation, or promotion evidence."
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
    parser.add_argument("--report", type=Path, default=DEFAULT_EVALUATION_REPORT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            generate_figure(args.report, args.figure, args.summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
