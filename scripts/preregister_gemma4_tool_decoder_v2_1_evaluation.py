#!/usr/bin/env python3
"""Seal the finite V2.1 local evaluation resource contract."""

from __future__ import annotations

import json

from semantic_3d_chat.evaluation.gemma4_tool_decoder_evaluation_preregistration_v2_1 import (
    write_evaluation_preregistration_v2_1,
)


def main() -> None:
    path, digest = write_evaluation_preregistration_v2_1()
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
