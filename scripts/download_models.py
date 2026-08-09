from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

from semantic_3d_chat.config import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--vision-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    selections = {"vision": config["vision"]}
    if not args.vision_only:
        selections["language"] = config["language"]
    revisions: dict[str, dict[str, str]] = {}
    for role, selection in selections.items():
        model_id = selection["model_id"]
        requested_revision = selection.get("revision", "main")
        info = model_info(model_id, revision=requested_revision)
        resolved_revision = info.sha
        local_path = snapshot_download(repo_id=model_id, revision=resolved_revision)
        revisions[role] = {
            "model_id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "local_snapshot": str(Path(local_path).resolve()),
        }
        print(f"{role}: {model_id}@{resolved_revision}")
    output = PROJECT_ROOT / "reports" / "metrics" / "model_revisions.json"
    existing = json.loads(output.read_text()) if output.exists() else {}
    existing.update(revisions)
    output.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
