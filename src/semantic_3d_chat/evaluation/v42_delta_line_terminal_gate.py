"""Seal the negative V42 response screen and authorize one V43 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from semantic_3d_chat.config import PROJECT_ROOT

DEFAULT_SCREEN = Path(
    "reports/gemma4/metrics/v42_v41_retry1_update8_no_step_diagnostic.json"
)
DEFAULT_OUTPUT = Path("reports/gemma4/metrics/v42_delta_line_terminal_gate.json")
V42_SOURCE = Path("src/semantic_3d_chat/evaluation/v42_delta_line_screen.py")
V42_TEST = Path("tests/test_v42_delta_line_screen.py")
V41_TERMINAL = Path(
    "reports/gemma4/metrics/v41_retry1_update8_terminal_gate.json"
)
PROTECTED = Path(
    "reports/gemma4/metrics/"
    "training_selection_gemma4_color_mirror_full_vocab_v11_resume36.json"
)
V43_OUTPUT = Path(
    "reports/gemma4/metrics/v43_aggregate_projected_no_step_diagnostic.json"
)

_PINS = {
    str(DEFAULT_SCREEN): "b325b8b4b1c616185dacc4a2f5d8f2f591b92f7f03e30a5968793a8edc2e6f32",
    str(V42_SOURCE): "96181438d5523a9df25e44446805357f0cbd6e8bb335af5c760eb2e8f5f6eb7f",
    str(V42_TEST): "9bb0ee7ba722b43b09af3f601c8761c9fec14635036720366dd8889f945a84f8",
    str(V41_TERMINAL): "16cd37d91ceb911904737d8b306a308c9a12984fd13d78ee10842f36c8b771fd",
    str(PROTECTED): "c0086f66edbb8854a7938e09c57535bfd47100adbaf3b3c95eeb4b08014ce2f8",
}
_ALPHAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
_V43_STEPS = [-0.008, -0.004, 0.0, 0.002, 0.004, 0.008, 0.012, 0.016]
_TARGET_U0 = "d0834cc588ee2a9edf08aabedfd01e0a6d2b01c6b6ae7e3a3d764eaddf58cc3e"
_FULL_U0 = "7b951c6d7ae4f7b50603159f0bc4dfb4d50b5b40f9325134d78d1de1dae87fc0"
_FROZEN = "cec01bc088bb87c6bb44e0659eb03aa766f951ddeee706ca9a70edaa080dea5e"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def build_report() -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in _PINS.items():
        path = _resolve(relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"V42 terminal input missing or aliased: {relative}")
        digest = _sha256(path)
        if digest != expected:
            raise ValueError(f"V42 terminal input changed: {relative}")
        observed[relative] = digest
    screen = json.loads(_resolve(DEFAULT_SCREEN).read_text(encoding="utf-8"))
    candidates = screen.get("candidate_results")
    restorations = screen.get("restoration_audit")
    endpoint = _mapping(screen.get("endpoint_replay"), "V42 endpoint replay")
    final_state = _mapping(screen.get("final_state"), "V42 final state")
    inventory = _mapping(screen.get("candidate_inventory"), "V42 inventory")
    if (
        screen.get("artifact") != "v42_v41_retry1_update8_no_step_diagnostic"
        or screen.get("screen_integrity_passed") is not True
        or screen.get("teacher_eligible_candidate_found") is not False
        or screen.get("selected_alpha") is not None
        or screen.get("selected_candidate") is not None
        or screen.get("selected_target_sha256") is not None
        or screen.get("validation_qa_loaded") is not False
        or screen.get("oracle_loaded") is not False
        or screen.get("final_test_scenes_touched") is not False
        or screen.get("forbidden_file_accesses") != []
        or screen.get("optimizer_constructed_or_loaded") is not False
        or screen.get("gradient_measurement_performed") is not False
        or screen.get("candidate_checkpoint_written") is not False
        or not isinstance(candidates, list)
        or len(candidates) != 9
        or [row.get("alpha") for row in candidates] != _ALPHAS
        or any(row.get("teacher_eligible") is not False for row in candidates)
        or not isinstance(restorations, list)
        or len(restorations) != 9
        or any(row.get("passed") is not True for row in restorations)
        or endpoint.get("passed") is not True
        or final_state.get("restored_exact") is not True
        or final_state.get("target_state_sha256") != _TARGET_U0
        or final_state.get("full_state_sha256") != _FULL_U0
        or final_state.get("frozen_state_sha256") != _FROZEN
        or inventory.get("fixed_alpha_grid") != _ALPHAS
        or inventory.get("adaptive_refinement") is not False
    ):
        raise ValueError("V42 negative-screen evidence changed")
    authorization = {
        "schema_version": 1,
        "authorization_id": "v43_aggregate_projected_train_only_no_step_screen",
        "authorized": True,
        "only_exact_action": "one_v43_aggregate_projected_response_screen",
        "authorized_output": str(V43_OUTPUT),
        "source_target_state_sha256": _TARGET_U0,
        "source_full_state_sha256": _FULL_U0,
        "source_frozen_state_sha256": _FROZEN,
        "gradient_surface": {
            "target_tensor": "layer14_q_proj_lora_b_only",
            "target_parameter_count": 16_384,
            "broad_component": "mean_48_unchanged_rows_times_1",
            "answer_component": "mean_8_priority_pair_answer_nll_times_0.5",
            "side_component": "mean_8_priority_pair_side_hinge_times_8",
            "cross_component": "mean_all_25_pair_cross_hinge_times_56",
            "autograd_api": "torch.autograd.grad",
            "parameter_grad_accumulation_authorized": False,
            "optimizer_authorized": False,
        },
        "projection": {
            "implementation": "v41_cpu_float64_active_set_qp",
            "scalar_clip_norm": 1.0,
            "candidate_formula": "float32(B0 - scalar_step * clipped_projected_gradient)",
            "fixed_scalar_steps": list(_V43_STEPS),
            "candidate_hashes_must_be_fixed_before_candidate_forward_evaluation": True,
            "adaptive_refinement_authorized": False,
        },
        "diagnostic_scope": {
            "training_qa_and_maps_only": True,
            "all_25_pair_and_48_broad_rows_per_candidate": True,
            "temporary_target_substitution_authorized": True,
            "exact_u0_restoration_after_every_candidate": True,
            "optimizer_construction_or_load_authorized": False,
            "optimizer_step_authorized": False,
            "checkpoint_write_authorized": False,
            "validation_access_authorized": False,
            "oracle_access_authorized": False,
            "final_test_access_authorized": False,
            "selector_execution_authorized": False,
            "runtime_promotion_authorized": False,
        },
        "new_terminal_seal_required_after_screen": True,
    }
    return {
        "schema_version": 1,
        "artifact": "v42_delta_line_terminal_gate",
        "passed": True,
        "screen_sha256": _PINS[str(DEFAULT_SCREEN)],
        "input_sha256": observed,
        "negative_result": {
            "fixed_alpha_grid": list(_ALPHAS),
            "candidate_count": 9,
            "teacher_eligible_candidate_count": 0,
            "endpoint_replay_exact": True,
            "all_candidates_restored_u0": True,
            "no_optimizer_gradient_checkpoint_or_restricted_access": True,
        },
        "conditional_successor_authorization": authorization,
        "only_exact_successor_authorized": (
            "v43_aggregate_projected_train_only_no_step_screen"
        ),
        "v43_screen_authorized": True,
        "training_authorized": False,
        "validation_access_authorized": False,
        "selector_execution_authorized": False,
        "runtime_promotion_authorized": False,
    }


def write_report(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = build_report()
    path = _resolve(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_report(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_report", "write_report"]
