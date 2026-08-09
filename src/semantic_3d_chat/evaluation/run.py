"""Score saved JSONL predictions against evaluation-only QA references.

This command never loads a language model, scene map, or chat runtime.  A
separate inference job writes predictions; this process only joins opaque scene
and question IDs to the QA split and computes deterministic metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import artifact_root, load_config, reports_root
from semantic_3d_chat.evaluation.metrics import score_predictions


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {source}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise TypeError(f"Expected JSON object at {source}:{line_number}")
        records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def score_jsonl_files(
    references_path: str | Path,
    predictions_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    references_source = Path(references_path).expanduser().resolve()
    predictions_source = Path(predictions_path).expanduser().resolve()
    if not references_source.is_file():
        raise FileNotFoundError(f"Reference JSONL does not exist: {references_source}")
    if not predictions_source.is_file():
        raise FileNotFoundError(f"Prediction JSONL does not exist: {predictions_source}")
    references = load_jsonl(references_source)
    predictions = load_jsonl(predictions_source)
    metrics = score_predictions(references, predictions)
    report = {
        **metrics,
        "references_path": str(references_source),
        "references_sha256": _sha256(references_source),
        "predictions_path": str(predictions_source),
        "predictions_sha256": _sha256(predictions_source),
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        _atomic_json(destination, report)
        report["metrics_path"] = str(destination)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--references", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    configured_reports_root = reports_root(config)
    references = args.references or artifact_root(config, "qa") / "test.jsonl"
    predictions = args.predictions or configured_reports_root / "predictions" / "test.jsonl"
    output = args.output or configured_reports_root / "metrics" / "metrics.json"
    report = score_jsonl_files(references, predictions, output_path=output)
    print(
        json.dumps(
            {
                "metrics_path": report["metrics_path"],
                "reference_count": report["reference_count"],
                "prediction_count": report["prediction_count"],
                "normalized_exact_accuracy": report["normalized_exact_accuracy"],
                "spatial_relation_accuracy": report["spatial_relation_accuracy"],
                "count_accuracy": report["count"]["accuracy"],
                "counterfactual": report["counterfactual"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
