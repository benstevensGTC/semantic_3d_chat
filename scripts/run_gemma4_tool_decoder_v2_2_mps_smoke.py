#!/usr/bin/env python3
"""Run the one released V2.2 MPS microbatch and atomically save evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from semantic_3d_chat.config import PROJECT_ROOT, load_config
from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_2 import (
    MPS_SMOKE_RELEASE_PATH,
    MPS_SMOKE_REPORT_PATH,
    sha256_file,
)
from semantic_3d_chat.training.train_gemma4_tool_decoder_v2 import (
    run_full_model_mps_microbatch_smoke_v2,
)


def _atomic_create(path: Path, payload: dict[str, object]) -> str:
    if path.exists():
        raise FileExistsError("V2.2 MPS smoke report is create-once")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return sha256_file(path)


def main() -> None:
    config = load_config("configs/experiments/gemma4_embodied_tool_decoder_v2.yaml")
    report = run_full_model_mps_microbatch_smoke_v2(
        config, authorization=MPS_SMOKE_RELEASE_PATH
    )
    output = PROJECT_ROOT / MPS_SMOKE_REPORT_PATH
    digest = _atomic_create(output, report)
    print(json.dumps({"path": str(output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
