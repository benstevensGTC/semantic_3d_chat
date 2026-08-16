"""Execute the reporting-only V3 amendment over the unchanged V2 protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

from semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v3_preregistration import (
    ARTIFACT,
    OUTPUT_CHECKPOINT,
    PREREGISTRATION,
    RESULT_REPORT,
    SMOKE_REPORT,
    V2_SMOKE,
    V2_SMOKE_SHA256,
    authenticate_preregistration,
)
from semantic_3d_chat.training import train_fixed_prefix_ple_v54 as v1


def canonical_tuple_mapping_hash(value: Mapping[tuple[str, str], float]) -> str:
    """Hash tuple-key diagnostics without affecting any numeric value."""

    records = [
        {"key": [first, second], "value": number}
        for (first, second), number in sorted(value.items())
    ]
    return v1._canonical_hash(records)


def _activate_v3_paths() -> None:
    v1.ARTIFACT = ARTIFACT
    v1.PREREGISTRATION = PREREGISTRATION
    v1.SMOKE_REPORT = SMOKE_REPORT
    v1.RESULT_REPORT = RESULT_REPORT
    v1.OUTPUT_CHECKPOINT = OUTPUT_CHECKPOINT
    v1.authenticate_preregistration = authenticate_preregistration


def _patched_teacher_forcing(bundle: Any, rows: Sequence[v1.ReaderRecord]) -> dict[str, Any]:
    """Exact V1 measurement with only tuple-key report hashing repaired."""

    installation = bundle.installation
    installation.eval()
    correct: dict[tuple[str, str], float] = {}
    batch_size = 2
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset : offset + batch_size]
        nlls = v1.answer_nlls(
            bundle,
            tuple((bundle.prefixes[row.scene_id], row) for row in selected),
        )
        for row, number in zip(selected, nlls.tolist(), strict=True):
            correct[(row.scene_id, row.question_id)] = float(number)
    changed = [row for row in rows if row.changed]
    wrong: dict[tuple[str, str], float] = {}
    for offset in range(0, len(changed), batch_size):
        selected = changed[offset : offset + batch_size]
        if any(row.paired_scene_id is None for row in selected):
            raise ValueError("PLE-V54 V3 validation changed row lacks paired scene")
        nlls = v1.answer_nlls(
            bundle,
            tuple((bundle.prefixes[str(row.paired_scene_id)], row) for row in selected),
        )
        for row, number in zip(selected, nlls.tolist(), strict=True):
            wrong[(row.scene_id, row.question_id)] = float(number)
    margins = {key: wrong[key] - correct[key] for key in wrong}
    units: dict[tuple[str, str], list[float]] = {}
    for row in changed:
        assert row.pair_id is not None and row.pair_question_key is not None
        units.setdefault((row.pair_id, row.pair_question_key), []).append(
            margins[(row.scene_id, row.question_id)]
        )
    if len(units) != 26 or any(len(values) != 2 for values in units.values()):
        raise ValueError("PLE-V54 V3 validation changed-unit inventory changed")
    positive = sum(number > 0.0 for number in margins.values())
    complete = sum(all(number > 0.0 for number in values) for values in units.values())
    return {
        "answer_nll_mean": sum(correct.values()) / len(correct),
        "answer_nll_count": len(correct),
        "changed_margin_mean": sum(margins.values()) / len(margins),
        "changed_positive_margin_sides": positive,
        "changed_side_count": len(margins),
        "changed_positive_margin_rate": positive / len(margins),
        "changed_complete_units": complete,
        "changed_unit_count": len(units),
        "correct_nll_sha256": canonical_tuple_mapping_hash(correct),
        "changed_margin_sha256": canonical_tuple_mapping_hash(margins),
    }


def _activate_measurement_patch() -> None:
    _activate_v3_paths()
    v1.evaluate_teacher_forcing = _patched_teacher_forcing


def structural_preflight() -> dict[str, Any]:
    _activate_measurement_patch()
    result = v1.structural_preflight()
    result["artifact"] = ARTIFACT
    result["v3_only_change"] = {
        "field": "diagnostic_hash_serialization.for_tuple_keyed_mappings",
        "affects_numeric_measurement": False,
    }
    return result


def inherit_smoke() -> dict[str, Any]:
    """Create a no-model V3 attestation to the exact passing V2 smoke."""

    _activate_v3_paths()
    if v1._resolve(SMOKE_REPORT).exists():
        raise FileExistsError("PLE-V54 V3 smoke attestation already exists")
    preregistration = authenticate_preregistration(PREREGISTRATION)
    if v1.sha256_file(V2_SMOKE) != V2_SMOKE_SHA256:
        raise ValueError("PLE-V54 V2 smoke bytes changed")
    source = v1._read_json(V2_SMOKE)
    if source.get("passed") is not True:
        raise ValueError("PLE-V54 V3 requires a passing V2 gradient smoke")
    report = {
        "schema_version": 1,
        "artifact": f"{ARTIFACT}_inherited_gradient_smoke",
        "status": "passed_by_exact_v2_smoke_inheritance",
        "passed": True,
        "source_path": V2_SMOKE,
        "source_sha256": V2_SMOKE_SHA256,
        "source_gradient_l2": source["gradient_l2"],
        "source_memory": source["memory"],
        "source_trainable_parameter_count": source["trainable_parameter_count"],
        "new_model_forward_executed": False,
        "reason": "V3 changes only post-forward diagnostic tuple-key serialization",
        "preregistration_sha256": preregistration["sha256"],
        "question_dependent_scene_retrieval": False,
        "environmental_text_inputs": [],
    }
    v1._atomic_create_json(SMOKE_REPORT, report)
    return report


def train_and_gate() -> dict[str, Any]:
    _activate_measurement_patch()
    return v1.train_and_gate()


def authenticate_result() -> dict[str, Any]:
    _activate_measurement_patch()
    return v1.authenticate_result()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "inherit-smoke", "train", "authenticate")
    )
    mode = parser.parse_args(argv).mode
    result = {
        "preflight": structural_preflight,
        "inherit-smoke": inherit_smoke,
        "train": train_and_gate,
        "authenticate": authenticate_result,
    }[mode]()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
