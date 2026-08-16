#!/usr/bin/env python3
"""Release V2.2 multi-update training only after the sealed MPS smoke passes."""

from __future__ import annotations

import json

from semantic_3d_chat.evaluation.gemma4_tool_decoder_training_authorization_v2_2 import (
    write_training_release_v2_2,
)


def main() -> None:
    path, digest = write_training_release_v2_2()
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
