#!/usr/bin/env python3
"""Seal bounded V2 evaluation rows before any heavy launch."""

from __future__ import annotations

import json

from semantic_3d_chat.evaluation.gemma4_tool_decoder_evaluation_preregistration_v2 import (
    write_evaluation_preregistration_v2,
)


def main() -> None:
    path, digest = write_evaluation_preregistration_v2()
    print(
        json.dumps(
            {"path": str(path), "sha256": digest, "status": "sealed"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
