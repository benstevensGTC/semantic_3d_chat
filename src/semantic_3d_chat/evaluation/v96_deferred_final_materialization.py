"""Execute one sealed deferred-materialization stage after V96 authorization.

The physical child commands and output paths come from V95's already-sealed,
outcome-independent preregistration.  Seven child commands are byte-identical;
the QA-selection child alone uses the V96 authorization repair while reusing
the unchanged pure selector.  Every stage requires the immutable V96 fixed
final, authenticated passing V96 gate, pre-materialization evaluator seal,
explicit create-once V96 unlock, intact wrapper sources, exact predecessor
receipts, and absent unreceipted outputs before any subprocess can start.

Importing this module and its ``preflight`` command execute no child process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation.v85_strict_multiscene_preflight import (
    sha256_file_v85,
)
from semantic_3d_chat.evaluation.v95_deferred_final_materialization import (
    RECEIPT_ROOT,
    authenticate_materialization_preregistration_v95,
)
from semantic_3d_chat.evaluation.v95_deferred_final_qa import FINAL_QA
from semantic_3d_chat.evaluation.v96_deferred_final import (
    CONFIG,
    MATERIALIZATION_STAGE_ORDER,
    _authenticate_deferred_final_unlock_under_guard_v96,
    _validate_materialization_preregistration_v96,
    deferred_final_guard_v96,
)
from semantic_3d_chat.evaluation.v96_deferred_final_evaluation import (
    authenticate_preregistration_v96_final,
)
from semantic_3d_chat.evaluation.v96_known_development_common_v2 import (
    canonical_sha256_v96,
    read_json_strict_v96,
    write_json_create_once_v96,
)

SCHEMA_VERSION: Final[int] = 96
RECEIPT_ARTIFACT: Final[str] = "gemma4_v96_deferred_final_stage_receipt_v1"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"V96 materialization path escapes project: {path}") from error


def _contract_path(raw: str, label: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"V96 {label} must be a project-relative fixed path: {raw}")
    root = PROJECT_ROOT.resolve()
    path = PROJECT_ROOT / candidate
    current = PROJECT_ROOT
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"V96 {label} traverses a symbolic link: {raw}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"V96 {label} escapes the project: {raw}") from error
    return resolved


def _stage_contract(materialization: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    if stage not in MATERIALIZATION_STAGE_ORDER:
        raise ValueError(f"Unknown V96 materialization stage: {stage}")
    _validate_materialization_preregistration_v96(materialization)
    contract = materialization["stages"][stage]
    if not isinstance(contract, Mapping):
        raise TypeError(f"V96 materialization contract is missing: {stage}")
    return contract


def _receipt_path(materialization: Mapping[str, Any], stage: str) -> Path:
    contract = _stage_contract(materialization, stage)
    path = _contract_path(str(contract["receipt"]), "receipt")
    expected = (RECEIPT_ROOT / f"{stage}.json").resolve()
    if path != expected:
        raise ValueError(f"V96 receipt path changed: {stage}")
    return path


def _output_paths(materialization: Mapping[str, Any], stage: str) -> tuple[Path, ...]:
    contract = _stage_contract(materialization, stage)
    return tuple(
        _contract_path(str(raw), f"{stage} output") for raw in contract["expected_outputs"]
    )


def _effective_child_argv(materialization: Mapping[str, Any], stage: str) -> list[list[str]]:
    """Return the sealed V95 commands except the V96 authorization repair."""

    contract = _stage_contract(materialization, stage)
    original = contract["child_argv"]
    if stage != "qa_select":
        return [list(argv) for argv in original]
    support_python = original[0][0]
    return [
        [
            support_python,
            "-m",
            "semantic_3d_chat.evaluation.v96_deferred_final_qa",
            "select",
            "--config",
            str(CONFIG),
        ]
    ]


def _assert_outputs_absent(materialization: Mapping[str, Any], stage: str) -> None:
    contract = _stage_contract(materialization, stage)
    paths = _output_paths(materialization, stage)
    present = [
        raw
        for raw, path in zip(contract["expected_outputs"], paths, strict=True)
        if (path.exists() or path.is_symlink())
        and not (
            path.resolve() == FINAL_QA.resolve()
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == 0
        )
    ]
    if present:
        raise FileExistsError(f"V96 refuses preexisting unreceipted outputs for {stage}: {present}")


def _existing_output_identity(materialization: Mapping[str, Any], stage: str) -> dict[str, str]:
    contract = _stage_contract(materialization, stage)
    paths = _output_paths(materialization, stage)
    identities: dict[str, str] = {}
    missing: list[str] = []
    for raw, path in zip(contract["expected_outputs"], paths, strict=True):
        if path.is_symlink() or not path.is_file():
            missing.append(raw)
        else:
            identities[raw] = sha256_file_v85(path)
    if missing:
        raise FileNotFoundError(f"V96 stage outputs are incomplete: {missing}")
    return identities


def _receipt_payload(
    *,
    materialization: Mapping[str, Any],
    unlock: Mapping[str, Any],
    stage: str,
    output_sha256: Mapping[str, str],
) -> dict[str, Any]:
    contract = _stage_contract(materialization, stage)
    payload = {
        "artifact": RECEIPT_ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": "completed_after_authenticated_v96_unlock",
        "preregistration_file_sha256": materialization["preregistration_file_sha256"],
        "preregistration_identity_sha256": materialization["preregistration_identity_sha256"],
        "unlock_file_sha256": unlock["unlock_file_sha256"],
        "unlock_identity_sha256": unlock["unlock_identity_sha256"],
        "candidate_fingerprint_sha256": unlock["candidate_fingerprint_sha256"],
        "implementation_source_inventory_sha256": unlock["implementation_source_inventory_sha256"],
        "child_argv": _effective_child_argv(materialization, stage),
        "original_v95_child_argv_sha256": canonical_sha256_v96(contract["child_argv"]),
        "v96_authorization_override": stage == "qa_select",
        "expected_outputs": contract["expected_outputs"],
        "output_sha256": dict(output_sha256),
        "output_inventory_sha256": canonical_sha256_v96(output_sha256),
        "stage_execution_performed": True,
        "automatic_runtime_promotion": False,
    }
    payload["receipt_identity_sha256"] = canonical_sha256_v96(payload)
    return payload


def _authenticate_receipt(
    materialization: Mapping[str, Any],
    unlock: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    path = _receipt_path(materialization, stage)
    actual = read_json_strict_v96(path)
    output_sha256 = _existing_output_identity(materialization, stage)
    expected = _receipt_payload(
        materialization=materialization,
        unlock=unlock,
        stage=stage,
        output_sha256=output_sha256,
    )
    if actual != expected:
        raise ValueError(f"V96 stage receipt or output changed: {stage}")
    return {
        **actual,
        "receipt_file_sha256": sha256_file_v85(path),
        "authenticated": True,
    }


def _authenticate_predecessors(
    materialization: Mapping[str, Any],
    unlock: Mapping[str, Any],
    stage: str,
) -> None:
    index = MATERIALIZATION_STAGE_ORDER.index(stage)
    for predecessor in MATERIALIZATION_STAGE_ORDER[:index]:
        _authenticate_receipt(materialization, unlock, predecessor)


def _assert_no_future_receipts(materialization: Mapping[str, Any], stage: str) -> None:
    index = MATERIALIZATION_STAGE_ORDER.index(stage)
    present = [
        future
        for future in MATERIALIZATION_STAGE_ORDER[index + 1 :]
        if (
            _receipt_path(materialization, future).exists()
            or _receipt_path(materialization, future).is_symlink()
        )
    ]
    if present:
        raise RuntimeError(f"V96 later-stage receipts exist before {stage} completion: {present}")


def materialization_preflight_v96(
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Authenticate both authorizations without starting a child process."""

    with deferred_final_guard_v96(config_path):
        final_evaluation = authenticate_preregistration_v96_final()
        materialization = authenticate_materialization_preregistration_v95()
        _validate_materialization_preregistration_v96(materialization)
        unlock = _authenticate_deferred_final_unlock_under_guard_v96(config_path)
        return {
            "artifact": "gemma4_v96_deferred_final_stage_runner_preflight_v1",
            "schema_version": SCHEMA_VERSION,
            "status": "passed_no_materialization_stage_executed",
            "unlock_file_sha256": unlock["unlock_file_sha256"],
            "unlock_identity_sha256": unlock["unlock_identity_sha256"],
            "candidate_fingerprint_sha256": unlock["candidate_fingerprint_sha256"],
            "preregistration_file_sha256": materialization["preregistration_file_sha256"],
            "preregistration_identity_sha256": materialization["preregistration_identity_sha256"],
            "final_evaluation_preregistration_file_sha256": final_evaluation[
                "preregistration_file_sha256"
            ],
            "final_evaluation_preregistration_identity_sha256": final_evaluation[
                "preregistration_identity_sha256"
            ],
            "stage_order": list(MATERIALIZATION_STAGE_ORDER),
            "stage_execution_performed": False,
            "child_process_started": False,
            "model_loaded": False,
            "automatic_runtime_promotion": False,
        }


def run_materialization_stage_v96(
    stage: str,
    config_path: str | Path = CONFIG,
) -> dict[str, Any]:
    """Run exactly one sealed child stage while holding the V96 process lock."""

    if stage not in MATERIALIZATION_STAGE_ORDER:
        raise ValueError(f"Unknown V96 materialization stage: {stage}")
    with deferred_final_guard_v96(config_path):
        # Both immutable authorizations are checked before receipt/output reads,
        # and therefore necessarily before any subprocess can be created.
        authenticate_preregistration_v96_final()
        materialization = authenticate_materialization_preregistration_v95()
        _validate_materialization_preregistration_v96(materialization)
        unlock = _authenticate_deferred_final_unlock_under_guard_v96(config_path)
        _stage_contract(materialization, stage)
        _authenticate_predecessors(materialization, unlock, stage)
        receipt_path = _receipt_path(materialization, stage)
        if receipt_path.exists() or receipt_path.is_symlink():
            receipt = _authenticate_receipt(materialization, unlock, stage)
            return {**receipt, "reused_authenticated_receipt": True}
        _assert_no_future_receipts(materialization, stage)
        _assert_outputs_absent(materialization, stage)
        environment = {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
        for argv in _effective_child_argv(materialization, stage):
            subprocess.run(
                argv,
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
            )
        output_sha256 = _existing_output_identity(materialization, stage)
        receipt = _receipt_payload(
            materialization=materialization,
            unlock=unlock,
            stage=stage,
            output_sha256=output_sha256,
        )
        write_json_create_once_v96(receipt_path, receipt)
        authenticated = _authenticate_receipt(materialization, unlock, stage)
        return {**authenticated, "reused_authenticated_receipt": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    run = subparsers.add_parser("run-stage")
    run.add_argument("--stage", choices=MATERIALIZATION_STAGE_ORDER, required=True)
    parser.add_argument("--config", default=str(CONFIG))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        materialization_preflight_v96(args.config)
        if args.command == "preflight"
        else run_materialization_stage_v96(args.stage, args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RECEIPT_ARTIFACT",
    "main",
    "materialization_preflight_v96",
    "run_materialization_stage_v96",
]
