#!/usr/bin/env python3
"""Seal V2.1 CPU inputs while denying both heavy stages."""

from __future__ import annotations

import json

from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_1 import (
    write_cpu_authorization_v2_1,
)


def main() -> None:
    path, digest = write_cpu_authorization_v2_1()
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
