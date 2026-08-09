"""Explicitly download the pinned Gemma 4 E2B production files."""

from __future__ import annotations

import json

from huggingface_hub import HfApi, snapshot_download

from semantic_3d_chat.vision.gemma4_probe import GEMMA4_MODEL_ID, GEMMA4_REVISION

ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
)


def main() -> None:
    info = HfApi().model_info(GEMMA4_MODEL_ID, revision=GEMMA4_REVISION)
    if info.sha != GEMMA4_REVISION:
        raise RuntimeError(f"Pinned revision resolved to {info.sha}, expected {GEMMA4_REVISION}")
    snapshot = snapshot_download(
        repo_id=GEMMA4_MODEL_ID,
        revision=GEMMA4_REVISION,
        allow_patterns=list(ALLOW_PATTERNS),
        max_workers=1,
    )
    print(
        json.dumps(
            {
                "model_id": GEMMA4_MODEL_ID,
                "revision": GEMMA4_REVISION,
                "snapshot": snapshot,
                "files": list(ALLOW_PATTERNS),
                "next_step": "Production extraction and LM loading default to local_files_only/offline.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
