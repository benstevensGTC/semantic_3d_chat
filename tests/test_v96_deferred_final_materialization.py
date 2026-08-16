from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from semantic_3d_chat.evaluation import (
    v96_deferred_final_materialization as materialization,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _preregistration() -> dict[str, Any]:
    stages = {
        stage: {
            "child_argv": [["fixed-child", stage]],
            "expected_outputs": [f"outputs/{stage}.bin"],
            "receipt": f"receipts/{stage}.json",
        }
        for stage in materialization.MATERIALIZATION_STAGE_ORDER
    }
    return {
        "authenticated": True,
        "preregistration_file_sha256": _digest("prereg-file"),
        "preregistration_identity_sha256": _digest("prereg-identity"),
        "stage_order": list(materialization.MATERIALIZATION_STAGE_ORDER),
        "stages": stages,
    }


def _unlock() -> dict[str, Any]:
    return {
        "authenticated": True,
        "unlock_file_sha256": _digest("unlock-file"),
        "unlock_identity_sha256": _digest("unlock-identity"),
        "candidate_fingerprint_sha256": _digest("candidate"),
        "implementation_source_inventory_sha256": _digest("sources"),
    }


@contextmanager
def _unguarded(*_args: Any, **_kwargs: Any) -> Any:
    yield {"authenticated": True}


def _patch_stage_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = _preregistration()
    unlock = _unlock()

    def authenticate_preregistration() -> dict[str, Any]:
        events.append("preregistration")
        return prereg

    def authenticate_final_evaluation() -> dict[str, Any]:
        events.append("final_evaluation")
        return {
            "authenticated": True,
            "preregistration_file_sha256": _digest("final-evaluation-file"),
            "preregistration_identity_sha256": _digest("final-evaluation-identity"),
        }

    def authenticate_unlock(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        events.append("unlock")
        return unlock

    monkeypatch.setattr(materialization, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(materialization, "RECEIPT_ROOT", tmp_path / "receipts")
    monkeypatch.setattr(materialization, "deferred_final_guard_v96", _unguarded)
    monkeypatch.setattr(
        materialization,
        "authenticate_materialization_preregistration_v95",
        authenticate_preregistration,
    )
    monkeypatch.setattr(
        materialization,
        "authenticate_preregistration_v96_final",
        authenticate_final_evaluation,
    )
    monkeypatch.setattr(
        materialization,
        "_authenticate_deferred_final_unlock_under_guard_v96",
        authenticate_unlock,
    )
    monkeypatch.setattr(
        materialization,
        "_validate_materialization_preregistration_v96",
        lambda *_a: None,
    )
    return prereg, unlock


def _write_stage_output(tmp_path: Path, stage: str, value: str = "complete") -> None:
    path = tmp_path / f"outputs/{stage}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_stage_preflight_authenticates_both_seals_without_child_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)

    def reject_subprocess(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("preflight must not start a child")

    monkeypatch.setattr(materialization.subprocess, "run", reject_subprocess)
    result = materialization.materialization_preflight_v96("synthetic.yaml")

    assert events == ["final_evaluation", "preregistration", "unlock"]
    assert result["status"] == "passed_no_materialization_stage_executed"
    assert result["stage_execution_performed"] is False
    assert result["child_process_started"] is False


def test_run_stage_authenticates_prereg_and_unlock_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)

    def run(argv: list[str], **_kwargs: Any) -> None:
        events.append("child")
        assert argv == ["fixed-child", "generate"]
        _write_stage_output(tmp_path, "generate")

    monkeypatch.setattr(materialization.subprocess, "run", run)

    result = materialization.run_materialization_stage_v96("generate", "synthetic.yaml")

    assert events[:4] == ["final_evaluation", "preregistration", "unlock", "child"]
    assert result["authenticated"] is True
    assert result["reused_authenticated_receipt"] is False
    assert result["status"] == "completed_after_authenticated_v96_unlock"
    assert result["automatic_runtime_promotion"] is False
    receipt = json.loads((tmp_path / "receipts/generate.json").read_text())
    assert receipt["child_argv"] == [["fixed-child", "generate"]]
    assert receipt["v96_authorization_override"] is False


def test_qa_raw_cannot_start_without_final_evaluation_preregistration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)

    def blocked() -> dict[str, Any]:
        events.append("final_evaluation_blocked")
        raise FileNotFoundError("final evaluation preregistration")

    def reject_child(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("qa_raw child must not start")

    monkeypatch.setattr(materialization, "authenticate_preregistration_v96_final", blocked)
    monkeypatch.setattr(materialization.subprocess, "run", reject_child)

    with pytest.raises(FileNotFoundError, match="final evaluation"):
        materialization.run_materialization_stage_v96("qa_raw", "synthetic.yaml")
    assert events == ["final_evaluation_blocked"]


def test_qa_select_replaces_only_the_impossible_v95_unlock_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    prereg, _unlock_value = _patch_stage_runtime(monkeypatch, tmp_path, events)

    # Complete and receipt-authenticate every predecessor without running the
    # child under test.
    for predecessor in materialization.MATERIALIZATION_STAGE_ORDER:
        if predecessor == "qa_select":
            break
        _write_stage_output(tmp_path, predecessor)
        output_hashes = materialization._existing_output_identity(prereg, predecessor)
        receipt = materialization._receipt_payload(
            materialization=prereg,
            unlock=_unlock(),
            stage=predecessor,
            output_sha256=output_hashes,
        )
        path = tmp_path / f"receipts/{predecessor}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")

    observed: list[list[str]] = []

    def run(argv: list[str], **_kwargs: Any) -> None:
        observed.append(argv)
        _write_stage_output(tmp_path, "qa_select")

    monkeypatch.setattr(materialization.subprocess, "run", run)
    result = materialization.run_materialization_stage_v96("qa_select", "synthetic.yaml")

    assert len(observed) == 1
    assert observed[0][1:4] == [
        "-m",
        "semantic_3d_chat.evaluation.v96_deferred_final_qa",
        "select",
    ]
    assert "v95_deferred_final_qa" not in observed[0]
    assert result["v96_authorization_override"] is True
    assert result["child_argv"] == [observed[0]]


def test_existing_receipt_is_reused_only_after_full_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)

    def run(_argv: list[str], **_kwargs: Any) -> None:
        events.append("child")
        _write_stage_output(tmp_path, "generate")

    monkeypatch.setattr(materialization.subprocess, "run", run)
    created = materialization.run_materialization_stage_v96("generate", "synthetic.yaml")
    reused = materialization.run_materialization_stage_v96("generate", "synthetic.yaml")

    assert created["reused_authenticated_receipt"] is False
    assert reused["reused_authenticated_receipt"] is True
    assert events.count("child") == 1


def test_unreceipted_preexisting_output_blocks_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)
    _write_stage_output(tmp_path, "generate", "partial")

    def reject_child(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("child must not run")

    monkeypatch.setattr(materialization.subprocess, "run", reject_child)
    with pytest.raises(FileExistsError, match="unreceipted outputs"):
        materialization.run_materialization_stage_v96("generate", "synthetic.yaml")
    assert events == ["final_evaluation", "preregistration", "unlock"]


def test_only_fixed_zero_byte_qa_placeholder_is_treated_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preregistration = _preregistration()
    placeholder = tmp_path / "outputs/qa_select.bin"
    placeholder.parent.mkdir(parents=True)
    placeholder.touch()
    monkeypatch.setattr(materialization, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(materialization, "FINAL_QA", placeholder)
    monkeypatch.setattr(
        materialization, "_validate_materialization_preregistration_v96", lambda *_a: None
    )

    materialization._assert_outputs_absent(preregistration, "qa_select")
    placeholder.write_text("not empty", encoding="utf-8")
    with pytest.raises(FileExistsError, match="unreceipted outputs"):
        materialization._assert_outputs_absent(preregistration, "qa_select")


def test_tampered_predecessor_output_blocks_next_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)

    def run(argv: list[str], **_kwargs: Any) -> None:
        stage = argv[-1]
        _write_stage_output(tmp_path, stage)

    monkeypatch.setattr(materialization.subprocess, "run", run)
    materialization.run_materialization_stage_v96("generate", "synthetic.yaml")
    _write_stage_output(tmp_path, "generate", "tampered")

    with pytest.raises(ValueError, match="receipt or output changed"):
        materialization.run_materialization_stage_v96("render", "synthetic.yaml")
    assert not (tmp_path / "outputs/render.bin").exists()


def test_tampered_receipt_is_rejected_before_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)

    def run(_argv: list[str], **_kwargs: Any) -> None:
        _write_stage_output(tmp_path, "generate")

    monkeypatch.setattr(materialization.subprocess, "run", run)
    materialization.run_materialization_stage_v96("generate", "synthetic.yaml")
    receipt_path = tmp_path / "receipts/generate.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["automatic_runtime_promotion"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt or output changed"):
        materialization.run_materialization_stage_v96("generate", "synthetic.yaml")


def test_missing_predecessor_receipt_blocks_out_of_order_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        materialization.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("child must not run")),
    )

    with pytest.raises(FileNotFoundError):
        materialization.run_materialization_stage_v96("render", "synthetic.yaml")


def test_future_receipt_blocks_earlier_uncompleted_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    _patch_stage_runtime(monkeypatch, tmp_path, events)
    future = tmp_path / "receipts/render.json"
    future.parent.mkdir(parents=True)
    future.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        materialization.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("child must not run")),
    )

    with pytest.raises(RuntimeError, match="Later-stage receipts|later-stage receipts"):
        materialization.run_materialization_stage_v96("generate", "synthetic.yaml")


def test_contract_path_rejects_absolute_and_parent_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(materialization, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="project-relative"):
        materialization._contract_path(str(tmp_path / "absolute"), "output")
    with pytest.raises(ValueError, match="project-relative"):
        materialization._contract_path("../escape", "output")


def test_contract_path_rejects_a_symbolic_link_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "outputs"
    linked.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(materialization, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="symbolic link"):
        materialization._contract_path("outputs/generate.bin", "output")


def test_stage_source_places_both_authentications_before_subprocess() -> None:
    source = inspect.getsource(materialization.run_materialization_stage_v96)
    assert source.index("authenticate_preregistration_v96_final") < source.index("subprocess.run")
    assert source.index("authenticate_materialization_preregistration_v95") < source.index(
        "subprocess.run"
    )
    assert source.index("_authenticate_deferred_final_unlock_under_guard_v96") < source.index(
        "subprocess.run"
    )
    assert source.index("_assert_outputs_absent") < source.index("subprocess.run")


def test_real_deferred_materialization_is_absent_or_a_valid_receipted_prefix() -> None:
    unlock = (
        materialization.PROJECT_ROOT
        / "reports/gemma4/metrics/gemma4_v96_atomic_pair_repair_deferred_final_unlock.json"
    )
    preregistration = materialization.authenticate_materialization_preregistration_v95()
    receipt_states = [
        materialization._receipt_path(preregistration, stage).is_file()
        for stage in materialization.MATERIALIZATION_STAGE_ORDER
    ]
    if unlock.exists() or unlock.is_symlink():
        authenticated_unlock = materialization._authenticate_deferred_final_unlock_under_guard_v96()
        seen_missing = False
        for stage, received in zip(
            materialization.MATERIALIZATION_STAGE_ORDER, receipt_states, strict=True
        ):
            if received:
                assert not seen_missing
                materialization._authenticate_receipt(preregistration, authenticated_unlock, stage)
            else:
                seen_missing = True
                materialization._assert_outputs_absent(preregistration, stage)
    else:
        assert not any(receipt_states)
        for stage in materialization.MATERIALIZATION_STAGE_ORDER:
            materialization._assert_outputs_absent(preregistration, stage)
