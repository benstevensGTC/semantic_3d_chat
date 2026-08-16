#!/usr/bin/env python3
"""Release exactly one full-model MPS backward smoke for V2.2."""

from __future__ import annotations

import json

from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_2 import (
    write_mps_smoke_release_v2_2,
)


def main() -> None:
    path, digest = write_mps_smoke_release_v2_2()
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
