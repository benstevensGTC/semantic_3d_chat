#!/usr/bin/env python3
"""Seal CPU inputs while explicitly denying Gemma/MPS training stages."""

from __future__ import annotations

import json

from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2 import (
    write_training_authorization_v2,
)


def main() -> None:
    path, digest = write_training_authorization_v2()
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
