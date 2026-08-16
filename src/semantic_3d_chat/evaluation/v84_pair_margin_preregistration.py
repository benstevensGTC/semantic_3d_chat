"""Seal the train-only V84 pair-margin follow-up before model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v84_strict_bridge_preflight import sha256_file_v84
from semantic_3d_chat.training.train_v84_pair_margin_followup import (
    CONFIG,
    PREREG_ARTIFACT,
    _scene_memories_v84,
    authenticate_pair_margin_sources_v84,
    load_pair_margin_config_v84,
    select_pair_margin_rows_v84,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _atomic_create_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = _resolve(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"V84 pair-margin preregistration exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_preregistration_v84(config_path: str | Path = CONFIG) -> dict[str, Any]:
    config = load_pair_margin_config_v84(config_path)
    source_hashes = authenticate_pair_margin_sources_v84(config)
    rows = select_pair_margin_rows_v84(config)
    memories, hashes = _scene_memories_v84(config, rows)
    parent = json.loads(
        _resolve(config["sources"]["parent_wiring_report"]).read_text(encoding="utf-8")
    )
    if not isinstance(parent, Mapping):
        raise TypeError("V84 parent wiring report must be an object")
    parent_final = parent.get("final_rows")
    if not isinstance(parent_final, list) or len(parent_final) != 2:
        raise ValueError("V84 parent wiring row inventory changed")
    parent_predictions = [row.get("greedy_prediction") for row in parent_final]
    parent_margins = [row.get("wrong_minus_correct_nll") for row in parent_final]
    if (
        parent.get("optimizer_updates") != 4
        or parent_predictions != ["under the table", "under the table"]
        or not all(isinstance(value, (int, float)) for value in parent_margins)
        or not any(float(value) <= 0.0 for value in parent_margins)
    ):
        raise ValueError("V84 follow-up rationale no longer matches the parent result")
    if any(tuple(memory.shape) != (1, 738, 1536) for memory in memories.values()):
        raise ValueError("V84 pair-margin prequestion memory shape changed")
    payload = {
        "artifact": PREREG_ARTIFACT,
        "schema_version": "84.1",
        "status": "sealed_before_first_followup_model_measurement",
        "config_path": Path(config_path).as_posix(),
        "config_sha256": sha256_file_v84(config_path),
        "training_source_path": config["sources"]["training_source"],
        "training_source_sha256": config["sources"]["training_source_sha256"],
        "authenticated_source_hashes": source_hashes,
        "rationale": {
            "parent_optimizer_updates": 4,
            "parent_same_greedy_prediction_both_scenes": len(set(parent_predictions)) == 1,
            "parent_has_nonpositive_correct_vs_wrong_margin": True,
            "parent_runtime_promoted": False,
        },
        "strict_input_contract": config["strict_input_contract"],
        "bridge": config["bridge"],
        "fixed_training_protocol": config["training"],
        "fixed_behavioral_gates": config["gates"],
        "wiring_unit": {
            "row_inventory": [[row.scene_id, row.question_id] for row in rows],
            "question_sha256": hashlib.sha256(rows[0].question.encode()).hexdigest(),
            "same_question_both_scenes": rows[0].question == rows[1].question,
            "answers_differ": rows[0].answer != rows[1].answer,
        },
        "scene_memories": {
            "compiled_before_question_tokenization": True,
            "shape_each": [1, 738, 1536],
            "hashes": hashes,
            "all_memory_slots_retained": True,
        },
        "full_gemma_model_loaded": False,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "development_behavior_scored": False,
        "sealed_historical_16_loaded": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
        "runtime_promotion_authorized": False,
    }
    output = _atomic_create_json(config["outputs"]["preregistration"], payload)
    payload["output"] = output.relative_to(PROJECT_ROOT).as_posix()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(argv)
    result = build_preregistration_v84(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
