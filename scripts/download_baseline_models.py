"""Download and pin evaluation-only baseline models without ONNX duplicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

from semantic_3d_chat.config import PROJECT_ROOT, load_config

CORE_PATTERNS = [
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
    "merges.txt",
    "tokenizer.*",
    "vocab.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    direct = config["evaluation"]["baselines"]["direct_multiview"]
    selections = {
        "oracle_text_language": config["language"],
        "direct_multiview": direct,
    }
    records: dict[str, dict[str, str]] = {}
    for role, selection in selections.items():
        model_id = str(selection["model_id"])
        requested_revision = str(selection.get("revision", "main"))
        info = model_info(model_id, revision=requested_revision)
        resolved_revision = str(info.sha)
        path = snapshot_download(
            repo_id=model_id,
            revision=resolved_revision,
            allow_patterns=CORE_PATTERNS,
        )
        license_name = None if info.card_data is None else info.card_data.license
        records[role] = {
            "license": str(license_name or selection.get("license", "unknown")),
            "local_snapshot": str(Path(path).resolve()),
            "model_id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
        }
        print(f"{role}: {model_id}@{resolved_revision}")
    output = PROJECT_ROOT / "reports" / "metrics" / "model_revisions.json"
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    existing.update(records)
    output.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
