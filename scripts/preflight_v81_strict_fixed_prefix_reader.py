#!/usr/bin/env python3
"""Run the sealed V81 model-free CPU preflight; never load Gemma or fit."""

from semantic_3d_chat.evaluation.v81_structured_dense_atlas_sidecar_preregistration import (
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
