"""Combine per-condition QA scores into the report's ablation artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_CONDITIONS = (
    "primary",
    "empty_scene_prefix",
    "wrong_scene_prefix",
    "semantic_shuffle",
    "position_shuffle",
    "geometry_only",
    "semantics_without_xyz",
    "remove_rgb",
    "remove_normals",
)


def aggregate(input_directory: Path, output: Path) -> dict:
    results = {}
    for condition in DEFAULT_CONDITIONS:
        path = input_directory / f"{condition}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing scored control: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        results[condition] = {
            "normalized_exact_accuracy": payload["normalized_exact_accuracy"],
            "spatial_relation_accuracy": payload["spatial_relation_accuracy"],
            "count_accuracy": payload["count"]["accuracy"],
            "grounding_mean_error_m": payload["grounding"]["mean_coordinate_error_m"],
            "counterfactual_changed_rate": payload["counterfactual"]["changed_when_expected_rate"],
            "source": str(path),
        }
    primary = results["primary"]["normalized_exact_accuracy"]
    for metrics in results.values():
        metrics["exact_accuracy_delta_vs_primary"] = metrics["normalized_exact_accuracy"] - primary
    artifact = {
        "schema_version": 1,
        "results": results,
        "interpretation": {
            "empty_prefix_collapse": results["empty_scene_prefix"]["normalized_exact_accuracy"]
            == 0.0,
            "content_ablation_max_abs_delta": max(
                abs(results[name]["exact_accuracy_delta_vs_primary"])
                for name in DEFAULT_CONDITIONS
                if name not in {"primary", "empty_scene_prefix"}
            ),
            "warning": (
                "A nonzero prefix is necessary, but the first multiscene checkpoint is "
                "insensitive to its scene-specific content."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("reports/metrics/controls"))
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/ablations.json"))
    args = parser.parse_args()
    artifact = aggregate(args.input_dir.resolve(), args.output.resolve())
    print(json.dumps(artifact["interpretation"], indent=2))


if __name__ == "__main__":
    main()
