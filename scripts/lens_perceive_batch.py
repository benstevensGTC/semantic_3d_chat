#!/usr/bin/env python3
"""Build point clouds for a set of rooms with one chosen model.

A cloud carries the output of one decoder's vision projector, so its width is
that decoder's hidden size and it cannot be read by a different model. Comparing
two models therefore means building two maps, not reusing one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.language.model_choice import add_model_arguments
from semantic_3d_chat.spatial_lens.grounding_data import available_rooms

GEMMA = PROJECT_ROOT / ".venv-gemma4" / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", type=int, default=25)
    parser.add_argument("--prefix", default="asset")
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--pixel-stride", type=int, default=6)
    add_model_arguments(parser)
    args = parser.parse_args()

    import random

    pool = [r for r in available_rooms() if r.startswith(args.prefix)]
    # The same shuffle the evaluations use, so "the first 25" means the same
    # twenty-five rooms everywhere.
    random.Random(20260818).shuffle(pool)
    chosen = sorted(pool[: args.rooms])

    built = skipped = failed = 0
    for index, room in enumerate(chosen, 1):
        destination = PROJECT_ROOT / "data" / "spatial_lens" / room / args.out_name
        if destination.is_file():
            skipped += 1
            continue
        started = time.time()
        done = subprocess.run(
            [
                str(GEMMA), str(PROJECT_ROOT / "scripts" / "lens_perceive.py"),
                "--room", room, "--model", args.model,
                "--out-name", args.out_name,
                "--pixel-stride", str(args.pixel_stride),
            ],
            capture_output=True, text=True, check=False, cwd=PROJECT_ROOT,
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        )
        if done.returncode != 0:
            sys.stderr.write(f"[{room}] failed:\n{done.stdout[-800:]}{done.stderr[-800:]}\n")
            failed += 1
            continue
        built += 1
        print(f"  {index}/{len(chosen)} {room}  {time.time() - started:.0f}s", flush=True)
    print(f"\nbuilt {built}, skipped {skipped}, failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
