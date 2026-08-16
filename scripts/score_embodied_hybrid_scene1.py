from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from semantic_3d_chat.evaluation.embodied_hybrid_oracle_score import (
    create_face_oracle_score,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-result",
        default="reports/gemma4/metrics/embodied_conversation_hybrid_scene_000001.json",
    )
    parser.add_argument("--scene-oracle", default="data/oracle/scene_000001/oracle.json")
    parser.add_argument(
        "--scoring-spec",
        default="configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/gemma4/metrics/"
            "embodied_conversation_hybrid_oracle_score_scene_000001.json"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = create_face_oracle_score(
        args.runtime_result,
        args.scene_oracle,
        args.scoring_spec,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
