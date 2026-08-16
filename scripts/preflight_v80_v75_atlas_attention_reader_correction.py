#!/usr/bin/env python3
"""Seal the corrective V80 CPU-preflight authenticator without loading Gemma."""

from semantic_3d_chat.evaluation.v80_cpu_preflight_correction import main

if __name__ == "__main__":
    raise SystemExit(main())
