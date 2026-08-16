"""Model-free, label-free authenticators for V96 evaluation evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from semantic_3d_chat.evaluation.v96_atomic_pair_repair_preflight import CONFIG
from semantic_3d_chat.evaluation.v96_known_development_common import (
    authenticate_nll_bundle_v96,
    authenticate_prediction_bundle_v96,
    authenticate_structured_score_v96,
)
from semantic_3d_chat.evaluation.v96_known_development_implementation import (
    hardened_evaluation_stage_v96,
)


@hardened_evaluation_stage_v96
def authenticate_v96(command: str, config: str) -> dict[str, Any]:
    if command == "prediction":
        bundle = authenticate_prediction_bundle_v96(config)
        return {
            "artifact": "gemma4_v96_prediction_bundle_authentication_v1",
            "authenticated": True,
            "row_count": len(bundle["rows"]),
            "candidate_fingerprint_sha256": bundle["fixed"].candidate[
                "fingerprint_sha256"
            ],
            "frozen_v95_state_sha256": bundle["fixed"].candidate[
                "frozen_v95_state_sha256"
            ],
            "prediction_sha256": bundle["prediction_sha256"],
            "prediction_access_sha256": bundle["access_sha256"],
            "prediction_completion_sha256": bundle["completion_sha256"],
            "labels_opened": False,
            "model_loaded": False,
            "runtime_promotion_authorized": False,
        }
    if command == "structured":
        result = authenticate_structured_score_v96(config)
        return {
            "artifact": "gemma4_v96_structured_score_authentication_v1",
            "authenticated": True,
            "structured_score_sha256": result["sha256"],
            "labels_opened": False,
            "model_loaded": False,
            "runtime_promotion_authorized": False,
        }
    if command == "nll":
        result = authenticate_nll_bundle_v96(config)
        return {
            "artifact": "gemma4_v96_nll_bundle_authentication_v1",
            "authenticated": True,
            "nll_sha256": result["sha256"],
            "nll_access_sha256": result["access_sha256"],
            "nll_completion_sha256": result["completion_sha256"],
            "labels_opened": False,
            "model_loaded": False,
            "runtime_promotion_authorized": False,
        }
    raise ValueError(f"Unknown V96 authentication command: {command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prediction", "structured", "nll"))
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(authenticate_v96(args.command, args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["authenticate_v96", "main"]
