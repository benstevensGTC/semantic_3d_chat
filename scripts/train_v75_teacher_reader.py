#!/usr/bin/env python3
"""Train the zero-safe nonlinear V75 verified-teacher scene reader."""

from __future__ import annotations

from train_v74_teacher_reader import main

if __name__ == "__main__":
    raise SystemExit(main(architecture_version="v75"))
