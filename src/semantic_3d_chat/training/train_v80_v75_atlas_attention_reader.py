"""Real zero-step smoke and bounded V80 reader screen.

Both phases use one local Gemma instance and native full-decoder forwards.  The
LM head materializes only answer-predecessor logits through Gemma's native
``logits_to_keep`` argument.  Every branch is prepared, forwarded, and released
sequentially; atlases and teacher logits remain on CPU between branches.

The gradient-smoke phase cannot construct an optimizer.  The bounded phase is
separately gated by a create-once passing smoke artifact and can perform only
the preregistered sixteen updates.  Neither phase contains a checkpoint writer.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import signal
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import psutil
import torch
import torch.nn.functional as F

from semantic_3d_chat.chat.file_audit import FileAccessAudit
from semantic_3d_chat.chat.question_control_runtime import _load_control_head
from semantic_3d_chat.chat.runtime import StaticChatRuntime
from semantic_3d_chat.chat.runtime_config import load_runtime_config
from semantic_3d_chat.config import PROJECT_ROOT
from semantic_3d_chat.evaluation import v75_fixed_atlas_behavior as v75b
from semantic_3d_chat.evaluation.v80_atlas_attention_reader_preregistration import (
    CONFIG,
    EXPECTED_CONFIG_SHA256,
    atomic_create_json,
    build_schedule_v80,
    load_v80_config,
    select_broad_held_v80,
    select_broad_train_v80,
    select_held_smoke_v80,
    sha256_file,
)
from semantic_3d_chat.evaluation.v80_cpu_preflight_correction import (
    CORRECTION as CPU_PREFLIGHT_CORRECTION,
)
from semantic_3d_chat.evaluation.v80_cpu_preflight_correction import validate_correction
from semantic_3d_chat.language.gemma4_answer_tail import answer_tail_forward
from semantic_3d_chat.language.local_lm import prompt_token_ids
from semantic_3d_chat.language.lora import tensor_state_sha256
from semantic_3d_chat.language.prefix_injection import (
    prefix_sha256,
    scene_boundary_mode_setting,
    scene_prefix_after_bos_setting,
)
from semantic_3d_chat.language.v80_atlas_attention_reader import (
    PARAMETER_COUNT,
    TARGET_MODULES,
    V80Installation,
    install_v80,
    validate_v80_topology,
)
from semantic_3d_chat.scene_encoder.fixed_prefix_atlas_v75 import (
    compile_fixed_scene_atlas_v75_v2,
)
from semantic_3d_chat.scene_encoder.question_control_v75 import (
    DenseFullSceneContinuousControlV75,
)
from semantic_3d_chat.training.train_adapter import tokenize_answer
from semantic_3d_chat.training.train_question_control_v56 import assert_answer_only_labels
from semantic_3d_chat.training.train_question_control_v73 import (
    ChangedUnitV73,
    RowV73,
    changed_units_v73,
    load_config_v73,
    load_training_rows_v73,
    split_rows_v73,
)

PREREGISTRATION_SHA256: Final[str] = (
    "e44dc9aed1176cdfc30befe56d50e49a31f1638223a529a01bb086f5b3ea5894"
)
CPU_PREFLIGHT_SHA256: Final[str] = (
    "e56e4a8a4e9dc450988eb6d8e3788b2469f51433ab9c862d1f973c3608066c70"
)
CPU_PREFLIGHT_CORRECTION_SHA256: Final[str] = (
    "0f0d0183d4e6deed942465116305f5698d09717c5f233ef351445c828729c2cb"
)
REQUIRED_MPS_HIGH_WATERMARK_RATIO: Final[str] = "0.75"
REQUIRED_MPS_LOW_WATERMARK_RATIO: Final[str] = "0.70"


@dataclass
class V80Bundle:
    runtime: StaticChatRuntime
    prefixes: dict[str, torch.Tensor]
    train_rows: tuple[RowV73, ...]
    held_rows: tuple[RowV73, ...]
    schedule: tuple[tuple[ChangedUnitV73, ...], ...]
    held_smoke: tuple[ChangedUnitV73, ...]
    broad_train: tuple[RowV73, ...]
    broad_held: tuple[RowV73, ...]
    config: dict[str, Any]
    audit: FileAccessAudit

    @property
    def language(self) -> Any:
        return self.runtime.language


class WallTimeExceededV80(RuntimeError):
    """Raised by the preregistered hard wall timer."""


@contextmanager
def _wall_timer(seconds: int) -> Iterator[None]:
    previous: Any = None

    def expire(_signal: int, _frame: Any) -> None:
        raise WallTimeExceededV80("V80 exceeded its preregistered wall-time ceiling")

    if seconds > 0 and hasattr(signal, "SIGALRM"):
        previous = signal.signal(signal.SIGALRM, expire)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _memory_metrics() -> dict[str, int | None]:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024
    current = None
    driver = None
    if torch.backends.mps.is_available():
        current = int(torch.mps.current_allocated_memory())
        driver_method = getattr(torch.mps, "driver_allocated_memory", None)
        driver = None if driver_method is None else int(driver_method())
    virtual = psutil.virtual_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT))
    return {
        "host_total_bytes": int(virtual.total),
        "host_available_bytes": int(virtual.available),
        "disk_free_bytes": int(disk.free),
        "peak_process_rss_bytes": rss,
        "mps_current_allocated_bytes": current,
        "mps_driver_allocated_bytes": driver,
    }


def _assert_memory(config: Mapping[str, Any], *, phase: str) -> dict[str, int | None]:
    safety = config["memory_safety"]
    metrics = _memory_metrics()
    if metrics["host_total_bytes"] != int(safety["host_unified_memory_bytes"]):
        raise MemoryError(f"V80 host-memory identity changed during {phase}")
    if int(metrics["host_available_bytes"] or 0) < int(
        safety["minimum_host_available_bytes"]
    ):
        raise MemoryError(f"V80 host available memory fell below its floor during {phase}")
    driver = metrics["mps_driver_allocated_bytes"]
    if driver is not None and driver > int(safety["maximum_mps_driver_allocated_bytes"]):
        raise MemoryError(f"V80 MPS driver memory exceeded its ceiling during {phase}")
    if int(metrics["peak_process_rss_bytes"] or 0) > int(
        safety["maximum_process_rss_bytes"]
    ):
        raise MemoryError(f"V80 process RSS exceeded its ceiling during {phase}")
    return metrics


def _release_branch(config: Mapping[str, Any], *, phase: str) -> dict[str, int | None]:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return _assert_memory(config, phase=phase)


def _authenticate_seals(config: Mapping[str, Any]) -> dict[str, str]:
    outputs = config["outputs"]
    expected = {
        outputs["preregistration"]: PREREGISTRATION_SHA256,
        outputs["cpu_preflight"]: CPU_PREFLIGHT_SHA256,
    }
    observed: dict[str, str] = {}
    for path, digest in expected.items():
        current = sha256_file(path)
        if current != digest:
            raise ValueError(f"V80 create-once seal changed: {path}")
        observed[path] = current
    prereg = json.loads(_resolve(outputs["preregistration"]).read_text(encoding="utf-8"))
    preflight = json.loads(_resolve(outputs["cpu_preflight"]).read_text(encoding="utf-8"))
    if (
        prereg.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or prereg.get("training_executed") is not False
        or prereg.get("optimizer_constructed") is not False
        or preflight.get("passed") is not True
        or preflight.get("real_model", {}).get("optimizer_updates") != 0
    ):
        raise ValueError("V80 sealed preregistration/preflight contract failed")
    correction = validate_correction(
        CPU_PREFLIGHT_CORRECTION,
        expected_sha256=CPU_PREFLIGHT_CORRECTION_SHA256,
    )
    if (
        correction.get("correction", {}).get("corrected_value") is not False
        or correction.get("authoritative_real_model_state", {}).get(
            "optimizer_updates"
        )
        != 0
    ):
        raise ValueError("V80 corrected CPU-preflight update fact failed")
    observed[CPU_PREFLIGHT_CORRECTION] = CPU_PREFLIGHT_CORRECTION_SHA256
    observed.update(_authenticate_live_inputs(config))
    return observed


def _authenticate_live_inputs(config: Mapping[str, Any]) -> dict[str, str]:
    """Recheck every mutable launch input instead of trusting an old seal alone."""

    inputs = config["inputs"]
    expected = {
        inputs["source_v73_config"]: inputs["source_v73_config_sha256"],
        inputs["historical_training_qa"]: inputs["historical_training_qa_sha256"],
        inputs["runtime_config"]: inputs["runtime_config_sha256"],
        str(Path(inputs["base_checkpoint"]) / "adapter.safetensors"): inputs[
            "base_adapter_sha256"
        ],
        str(Path(inputs["atlas_controller"]) / "control.safetensors"): inputs[
            "atlas_controller_weights_sha256"
        ],
        str(Path(inputs["numeric_probe_bank"]) / "probes.safetensors"): inputs[
            "numeric_probe_file_sha256"
        ],
    }
    observed: dict[str, str] = {}
    for path, digest in expected.items():
        current = sha256_file(path)
        if current != digest:
            raise ValueError(f"V80 live launch input changed: {path}")
        observed[f"live:{path}"] = current

    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        / str(inputs["model_revision"])
    )
    model_blob = (snapshot / "model.safetensors").resolve(strict=True)
    if model_blob.name != inputs["model_file_sha256"] or not model_blob.is_file():
        raise ValueError("V80 content-addressed Gemma model blob changed")
    observed["live:gemma_model_blob_sha256_identity"] = model_blob.name
    return observed


def _assert_launch_environment() -> None:
    if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") != REQUIRED_MPS_HIGH_WATERMARK_RATIO:
        raise RuntimeError("V80 requires PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.75")
    if os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") != REQUIRED_MPS_LOW_WATERMARK_RATIO:
        raise RuntimeError("V80 requires PYTORCH_MPS_LOW_WATERMARK_RATIO=0.70")
    if not torch.backends.mps.is_available():
        raise RuntimeError("V80 real phases require MPS")


def _forbidden_roots() -> list[Path]:
    roots = [
        PROJECT_ROOT / "data" / "oracle",
        PROJECT_ROOT / "data" / "qa",
        PROJECT_ROOT / "configs" / "benchmarks" / "oracle",
        PROJECT_ROOT / "reports" / "gemma4" / "predictions" / "v75_official_validation.jsonl",
        PROJECT_ROOT / "reports" / "gemma4" / "metrics" / "v75_official_validation_score.json",
        PROJECT_ROOT / "reports" / "gemma4" / "questions",
        PROJECT_ROOT / "reports" / "gemma4" / "scorer_only",
    ]
    for scene_index in range(57, 63):
        scene_id = f"scene_{scene_index:06d}"
        roots.extend(
            PROJECT_ROOT / root / scene_id
            for root in (
                "data/maps",
                "data/features",
                "data/rendered",
                "data_gemma4/maps",
                "data_gemma4/features",
                "data_gemma4/rendered",
                "data_gemma4/scene_tokens",
            )
        )
    return roots


def _load_bundle(config: dict[str, Any], audit: FileAccessAudit) -> V80Bundle:
    inputs = config["inputs"]
    v73 = load_config_v73(inputs["source_v73_config"])
    rows = load_training_rows_v73(v73["training_qa"])
    train_rows, held_rows = split_rows_v73(rows)
    train_scenes = tuple(sorted({row.scene_id for row in train_rows}))
    held_scenes = tuple(sorted({row.scene_id for row in held_rows}))
    all_scenes = (*train_scenes, *held_scenes)

    probes, _probe_metadata = v75b._load_probe_bank(
        _resolve(inputs["numeric_probe_bank"]), audit
    )
    base_prefixes, _prefix_manifest = v75b._load_base_prefixes(
        _resolve(inputs["base_prefix_cache"]), all_scenes, audit
    )
    controller, controller_metadata = _load_control_head(
        _resolve(inputs["atlas_controller"]),
        hidden_size=1536,
        device=torch.device("cpu"),
        audit=audit,
    )
    if (
        type(controller) is not DenseFullSceneContinuousControlV75
        or controller_metadata.get("weights_sha256")
        != inputs["atlas_controller_weights_sha256"]
    ):
        raise ValueError("V80 exact V75 atlas controller changed")

    # Materialize every scene before the first prompt/question is prepared.
    prefixes: dict[str, torch.Tensor] = {}
    for scene_id in all_scenes:
        compiled = compile_fixed_scene_atlas_v75_v2(
            base_prefixes[scene_id], controller, probes
        )
        if (
            tuple(compiled.scene_prefix.shape) != (1, 738, 1536)
            or compiled.audit.question_dependent_scene_processing
            or compiled.audit.question_dependent_retrieval
            or compiled.audit.semantic_or_spatial_top_k_selection
            or not compiled.audit.base_environment_tokens_preserved_exactly
            or not compiled.audit.atlas_key_value_tokens_preserved_exactly
        ):
            raise RuntimeError("V80 compiled atlas failed its immutable contract")
        prefixes[scene_id] = compiled.scene_prefix.detach().cpu().contiguous()
    hashes_before = {scene: prefix_sha256(value) for scene, value in prefixes.items()}

    runtime_config_path = _resolve(inputs["runtime_config"])
    audit.record(runtime_config_path)
    runtime_config = load_runtime_config(runtime_config_path)
    runtime = StaticChatRuntime.load(
        runtime_config,
        train_scenes[0],
        checkpoint=_resolve(inputs["base_checkpoint"]),
        audit=audit,
        local_files_only=True,
    )
    if runtime.language.device.type != "mps" or runtime.language.backend_name != "gemma4":
        raise RuntimeError("V80 requires one local MPS Gemma-4 runtime")
    runtime.language.model.requires_grad_(False).eval()
    validate_v80_topology(runtime.language.model)
    decoder = runtime.language.decoder_module
    disable = getattr(decoder, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if bool(getattr(decoder, "is_gradient_checkpointing", False)):
        raise RuntimeError("V80 gradient checkpointing must remain disabled")
    hashes_after = {scene: prefix_sha256(value) for scene, value in prefixes.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("V80 atlas prefix changed before question preparation")
    return V80Bundle(
        runtime=runtime,
        prefixes=prefixes,
        train_rows=train_rows,
        held_rows=held_rows,
        schedule=build_schedule_v80(changed_units_v73(train_rows)),
        held_smoke=select_held_smoke_v80(held_rows),
        broad_train=select_broad_train_v80(train_rows),
        broad_held=select_broad_held_v80(held_rows),
        config=config,
        audit=audit,
    )


def _prepare(bundle: V80Bundle, scene_id: str, row: RowV73) -> Any:
    language = bundle.language
    backend = language.prefix_backend
    if backend is None:
        raise RuntimeError("V80 requires Gemma's native prefix backend")
    _assert_memory(bundle.config, phase="before_prepare")
    model_dtype = next(language.model.parameters()).dtype
    scene_prefix = bundle.prefixes[scene_id].to(
        device=language.device, dtype=model_dtype
    )
    prompt_ids = prompt_token_ids(
        language.tokenizer,
        str(bundle.runtime.config["language"]["system_prompt"]),
        row.question,
        language.device,
    )
    answer_ids = tokenize_answer(language.tokenizer, row.answer, language.device)
    prepared = backend.prepare(
        scene_prefix,
        prompt_ids,
        answer_ids,
        scene_prefix_after_bos=scene_prefix_after_bos_setting(bundle.runtime.config),
        scene_boundary_mode=scene_boundary_mode_setting(bundle.runtime.config),
    )
    assert_answer_only_labels(prepared.labels, answer_ids)
    return prepared


def _tail(bundle: V80Bundle, scene_id: str, row: RowV73) -> Any:
    prepared = _prepare(bundle, scene_id, row)
    tail = answer_tail_forward(bundle.language, prepared)
    del prepared
    return tail


def _nll_value(bundle: V80Bundle, scene_id: str, row: RowV73) -> float:
    with torch.no_grad():
        tail = _tail(bundle, scene_id, row)
        value = float(tail.mean_nll.detach().cpu())
        del tail
    _release_branch(bundle.config, phase="after_nll_value")
    return value


def _backward_nll(
    bundle: V80Bundle, scene_id: str, row: RowV73, coefficient: float
) -> float:
    if not math.isfinite(coefficient):
        raise ValueError("V80 NLL gradient coefficient is nonfinite")
    tail = _tail(bundle, scene_id, row)
    value = float(tail.mean_nll.detach().cpu())
    (tail.mean_nll * float(coefficient)).backward()
    del tail
    _release_branch(bundle.config, phase="after_backward_nll")
    return value


def _pair_metrics(
    bundle: V80Bundle, units: Sequence[ChangedUnitV73]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for unit in units:
        sides: list[dict[str, Any]] = []
        for row, wrong_scene in (
            (unit.left, unit.right.scene_id),
            (unit.right, unit.left.scene_id),
        ):
            correct = _nll_value(bundle, row.scene_id, row)
            wrong = _nll_value(bundle, wrong_scene, row)
            margin = wrong - correct
            sides.append(
                {
                    "scene_id": row.scene_id,
                    "wrong_scene_id": wrong_scene,
                    "correct_nll": correct,
                    "wrong_scene_nll": wrong,
                    "wrong_minus_correct_margin": margin,
                    "margin_softplus": float(F.softplus(torch.tensor(0.5 - margin))),
                }
            )
        records.append(
            {
                "pair_id": unit.pair_id,
                "question_key": unit.question_key,
                "change_type": unit.change_type,
                "sides": sides,
                "complete_unit": all(
                    side["wrong_minus_correct_margin"] > 0.0 for side in sides
                ),
            }
        )
    sides = [side for record in records for side in record["sides"]]
    families: defaultdict[str, list[float]] = defaultdict(list)
    for record in records:
        families[record["change_type"]].extend(
            side["wrong_minus_correct_margin"] for side in record["sides"]
        )
    return {
        "unit_count": len(records),
        "side_count": len(sides),
        "mean_correct_nll": sum(side["correct_nll"] for side in sides) / len(sides),
        "mean_wrong_scene_margin": sum(
            side["wrong_minus_correct_margin"] for side in sides
        )
        / len(sides),
        "mean_wrong_scene_softplus": sum(side["margin_softplus"] for side in sides)
        / len(sides),
        "positive_margin_sides": sum(
            side["wrong_minus_correct_margin"] > 0.0 for side in sides
        ),
        "complete_units": sum(record["complete_unit"] for record in records),
        "family_mean_margin": {
            family: sum(values) / len(values) for family, values in sorted(families.items())
        },
        "records": records,
    }


def _broad_metrics(bundle: V80Bundle, rows: Sequence[RowV73]) -> dict[str, Any]:
    values = [
        {"scene_id": row.scene_id, "question_id": row.question_id, "nll": _nll_value(bundle, row.scene_id, row)}
        for row in rows
    ]
    return {
        "row_count": len(values),
        "mean_answer_nll": sum(row["nll"] for row in values) / len(values),
        "records": values,
    }


def _state_sha256(installation: V80Installation) -> str:
    return tensor_state_sha256(installation.state_dict())


def softplus_nll_gradient_coefficients(
    *,
    correct_nll: float,
    wrong_nll: float,
    target: float,
    correct_weight: float,
    pair_weight: float,
    scale: float,
) -> tuple[float, float]:
    """Return exact branch coefficients for the preregistered causal loss.

    The loss is ``scale * (correct_weight * correct_nll + pair_weight *
    softplus(target - wrong_nll + correct_nll))``.  Keeping this scalar
    decomposition explicit lets the two full Gemma branches run sequentially
    without retaining two decoder graphs at once.
    """

    logit = target - wrong_nll + correct_nll
    if logit >= 0.0:
        sigmoid = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        sigmoid = exp_logit / (1.0 + exp_logit)
    return (
        scale * (correct_weight + pair_weight * sigmoid),
        scale * (-pair_weight * sigmoid),
    )


def _pair_gradient(
    bundle: V80Bundle, unit: ChangedUnitV73, *, scale: float
) -> dict[str, Any]:
    diagnostics: list[dict[str, float | str]] = []
    correct_weight = float(bundle.config["optimization"]["correct_answer_nll_weight"])
    target = float(bundle.config["optimization"]["margin_target_nats"])
    pair_weight = float(
        bundle.config["optimization"]["wrong_paired_scene_softplus_weight"]
    )
    for row, wrong_scene in (
        (unit.left, unit.right.scene_id),
        (unit.right, unit.left.scene_id),
    ):
        correct = _nll_value(bundle, row.scene_id, row)
        wrong = _nll_value(bundle, wrong_scene, row)
        correct_coefficient, wrong_coefficient = softplus_nll_gradient_coefficients(
            correct_nll=correct,
            wrong_nll=wrong,
            target=target,
            correct_weight=correct_weight,
            pair_weight=pair_weight,
            scale=scale,
        )
        _backward_nll(bundle, row.scene_id, row, correct_coefficient)
        _backward_nll(bundle, wrong_scene, row, wrong_coefficient)
        diagnostics.append(
            {
                "scene_id": row.scene_id,
                "wrong_scene_id": wrong_scene,
                "correct_nll": correct,
                "wrong_scene_nll": wrong,
                "margin": wrong - correct,
                "correct_gradient_coefficient": correct_coefficient,
                "wrong_gradient_coefficient": wrong_coefficient,
            }
        )
    return {"pair_id": unit.pair_id, "question_key": unit.question_key, "sides": diagnostics}


def _identity_check(
    bundle: V80Bundle, installation: V80Installation, baseline_logits: torch.Tensor, baseline_nll: float
) -> dict[str, Any]:
    row = bundle.held_smoke[0].left
    with torch.no_grad():
        tail = _tail(bundle, row.scene_id, row)
        wrapped_logits = tail.logits.detach().cpu().contiguous()
        wrapped_nll = float(tail.mean_nll.detach().cpu())
        del tail
    _release_branch(bundle.config, phase="after_identity_check")
    return {
        "answer_logits_bit_exact": torch.equal(baseline_logits, wrapped_logits),
        "answer_nll_bit_exact": baseline_nll == wrapped_nll,
        "baseline_nll": baseline_nll,
        "wrapped_nll": wrapped_nll,
        "maximum_logit_absolute_delta": float(
            (baseline_logits.float() - wrapped_logits.float()).abs().max()
        ),
        "parameter_count": installation.parameter_count,
    }


def _audit_summary(audit: FileAccessAudit) -> dict[str, Any]:
    return {
        "loaded_file_count": len(audit.unique_paths),
        "forbidden_access_count": len(audit.forbidden_accesses()),
        "forbidden_accesses": audit.forbidden_accesses(),
        "passed": not audit.forbidden_accesses(),
    }


def run_gradient_smoke(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Run the mandatory real zero-step gate; never construct an optimizer."""

    _assert_launch_environment()
    config = load_v80_config(config_path)
    seals = _authenticate_seals(config)
    output = _resolve(config["outputs"]["gradient_smoke"])
    prohibited = _resolve(config["outputs"]["prohibited_checkpoint"])
    if output.exists() or output.is_symlink() or prohibited.exists() or prohibited.is_symlink():
        raise FileExistsError("V80 gradient smoke already exists or checkpoint is present")
    started = time.perf_counter()
    audit = FileAccessAudit(_forbidden_roots(), block_forbidden=True)
    with _wall_timer(int(config["wall_time_budget_seconds"])), audit:
        _assert_memory(config, phase="before_load")
        bundle = _load_bundle(config, audit)
        after_load = _assert_memory(config, phase="after_load")

        # Held behavior is measured with the exact frozen atlas reader before
        # any new trainable parameter exists.
        baseline_held = _pair_metrics(bundle, bundle.held_smoke)
        baseline_broad = _broad_metrics(bundle, bundle.broad_held)
        identity_row = bundle.held_smoke[0].left
        with torch.no_grad():
            baseline_tail = _tail(bundle, identity_row.scene_id, identity_row)
            baseline_logits = baseline_tail.logits.detach().cpu().contiguous()
            baseline_nll = float(baseline_tail.mean_nll.detach().cpu())
            del baseline_tail
        _release_branch(config, phase="after_baseline_identity")

        installation = install_v80(bundle.language.model)
        installation.assert_fp32_finite()
        initial_state = _state_sha256(installation)
        identity = _identity_check(bundle, installation, baseline_logits, baseline_nll)
        installation.assert_only_adapters_trainable(bundle.language.model)
        for parameter in installation.parameters():
            parameter.grad = None

        selected = bundle.schedule[0]
        gradient_records = [
            _pair_gradient(bundle, unit, scale=1.0 / (2.0 * len(selected)))
            for unit in selected
        ]
        gradients = installation.gradient_norms()
        final_state = _state_sha256(installation)
        a_zero = all(item["residual_a"] == 0.0 for item in gradients.values())
        b_positive = all(
            item["residual_b"] is not None
            and math.isfinite(float(item["residual_b"]))
            and float(item["residual_b"]) > 0.0
            for item in gradients.values()
        )
        total_gradient = math.sqrt(
            sum(
                float(value or 0.0) ** 2
                for item in gradients.values()
                for value in item.values()
            )
        )
        final_memory = _assert_memory(config, phase="gradient_smoke_complete")
        checks = {
            "held_baseline_measured_before_adapter": True,
            "exact_zero_answer_logits": identity["answer_logits_bit_exact"],
            "exact_zero_answer_nll": identity["answer_nll_bit_exact"],
            "exact_zero_adapter_output": identity["maximum_logit_absolute_delta"] == 0.0,
            "trainable_parameter_count_exact": installation.parameter_count == PARAMETER_COUNT,
            "only_four_target_adapters_trainable": True,
            "all_four_a_gradients_exact_zero": a_zero,
            "all_four_b_gradients_finite_positive": b_positive,
            "finite_positive_total_gradient": math.isfinite(total_gradient)
            and total_gradient > 0.0,
            "adapter_state_unchanged_without_optimizer": initial_state == final_state,
            "optimizer_not_constructed": True,
            "zero_optimizer_updates": True,
            "checkpoint_absent": not prohibited.exists(),
            "memory_safety_passed": True,
        }
    audit.assert_clean()
    checks["file_audit_clean"] = True
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "artifact": "gemma4_v80_v75_atlas_attention_reader_gradient_smoke_v1",
        "status": "gradient_smoke_pass_training_still_not_launched"
        if passed
        else "gradient_smoke_fail_terminal_no_training",
        "passed": passed,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "sealed_inputs": seals,
        "implementation_sha256": {
            "training_module": sha256_file(__file__),
            "reader_module": sha256_file(
                "src/semantic_3d_chat/language/v80_atlas_attention_reader.py"
            ),
        },
        "checks": checks,
        "baseline_held_pair_metrics": baseline_held,
        "baseline_broad_held_metrics": baseline_broad,
        "zero_initialization": identity,
        "gradient_units": gradient_records,
        "gradient_norms": gradients,
        "total_gradient_l2": total_gradient,
        "target_modules": list(TARGET_MODULES),
        "initial_adapter_state_sha256": initial_state,
        "final_adapter_state_sha256": final_state,
        "optimizer_constructed": False,
        "optimizer_updates": 0,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
        "memory": {"after_model_load": after_load, "terminal": final_memory},
        "audit": _audit_summary(audit),
        "elapsed_seconds": time.perf_counter() - started,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
    }


def _pair_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean_correct_nll": float(after["mean_correct_nll"])
        - float(before["mean_correct_nll"]),
        "mean_wrong_scene_margin": float(after["mean_wrong_scene_margin"])
        - float(before["mean_wrong_scene_margin"]),
        "mean_wrong_scene_softplus": float(after["mean_wrong_scene_softplus"])
        - float(before["mean_wrong_scene_softplus"]),
        "positive_margin_sides": int(after["positive_margin_sides"])
        - int(before["positive_margin_sides"]),
        "complete_units": int(after["complete_units"]) - int(before["complete_units"]),
        "family_margin": {
            family: float(after["family_mean_margin"][family]) - float(value)
            for family, value in before["family_mean_margin"].items()
        },
    }


def _broad_retention_backward(
    bundle: V80Bundle,
    installation: V80Installation,
    row: RowV73,
) -> dict[str, float | bool]:
    with installation.disabled(), torch.no_grad():
        teacher_tail = _tail(bundle, row.scene_id, row)
        teacher = teacher_tail.logits.detach().cpu().float().contiguous()
        del teacher_tail
    _release_branch(bundle.config, phase="after_retention_teacher")
    candidate = _tail(bundle, row.scene_id, row)
    teacher_device = teacher.to(candidate.logits.device)
    teacher_prob = torch.softmax(teacher_device, dim=-1)
    kl = F.kl_div(
        torch.log_softmax(candidate.logits.float(), dim=-1),
        teacher_prob,
        reduction="batchmean",
    ).clamp_min(0.0)
    broad_weight = float(bundle.config["optimization"]["broad_answer_nll_weight"])
    retention_weight = float(bundle.config["optimization"]["retention_kl_weight"])
    loss = broad_weight * candidate.mean_nll + retention_weight * kl
    loss.backward()
    result = {
        "answer_nll": float(candidate.mean_nll.detach().cpu()),
        "kl_nats": float(kl.detach().cpu()),
        "answer_token_top1_agreement": bool(
            torch.equal(
                candidate.logits.detach().argmax(dim=-1).cpu(), teacher.argmax(dim=-1)
            )
        ),
    }
    del candidate, teacher_device, teacher_prob, teacher, kl, loss
    _release_branch(bundle.config, phase="after_broad_retention_backward")
    return result


def _retention_metrics(
    bundle: V80Bundle, installation: V80Installation, rows: Sequence[RowV73]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in rows:
        with installation.disabled(), torch.no_grad():
            teacher_tail = _tail(bundle, row.scene_id, row)
            teacher = teacher_tail.logits.detach().cpu().float().contiguous()
            del teacher_tail
        _release_branch(bundle.config, phase="after_retention_eval_teacher")
        with torch.no_grad():
            candidate = _tail(bundle, row.scene_id, row)
            current = candidate.logits.detach().cpu().float()
            del candidate
        probability = torch.softmax(teacher, dim=-1)
        kl = float(
            F.kl_div(
                torch.log_softmax(current, dim=-1), probability, reduction="batchmean"
            ).clamp_min(0.0)
        )
        records.append(
            {
                "scene_id": row.scene_id,
                "question_id": row.question_id,
                "kl_nats": kl,
                "answer_token_top1_agreement": torch.equal(
                    current.argmax(dim=-1), teacher.argmax(dim=-1)
                ),
            }
        )
        del teacher, current, probability
        _release_branch(bundle.config, phase="after_retention_eval_candidate")
    return {
        "row_count": len(records),
        "mean_kl_nats": sum(row["kl_nats"] for row in records) / len(records),
        "maximum_kl_nats": max(row["kl_nats"] for row in records),
        "answer_token_top1_agreement": sum(
            row["answer_token_top1_agreement"] for row in records
        )
        / len(records),
        "records": records,
    }


def run_bounded_screen(config_path: str | Path = CONFIG) -> dict[str, Any]:
    """Run at most sixteen updates after authenticating a passing smoke."""

    _assert_launch_environment()
    config = load_v80_config(config_path)
    _authenticate_seals(config)
    smoke_path = _resolve(config["outputs"]["gradient_smoke"])
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        smoke.get("passed") is not True
        or smoke.get("optimizer_constructed") is not False
        or smoke.get("optimizer_updates") != 0
        or smoke.get("config_sha256") != EXPECTED_CONFIG_SHA256
    ):
        raise ValueError("V80 bounded screen lacks a passing zero-step smoke")
    expected_implementation = smoke.get("implementation_sha256")
    current_implementation = {
        "training_module": sha256_file(__file__),
        "reader_module": sha256_file(
            "src/semantic_3d_chat/language/v80_atlas_attention_reader.py"
        ),
    }
    if expected_implementation != current_implementation:
        raise ValueError("V80 implementation changed after its gradient smoke")
    output = _resolve(config["outputs"]["bounded_screen"])
    prohibited = _resolve(config["outputs"]["prohibited_checkpoint"])
    if output.exists() or output.is_symlink() or prohibited.exists() or prohibited.is_symlink():
        raise FileExistsError("V80 bounded output/checkpoint already exists")

    started = time.perf_counter()
    audit = FileAccessAudit(_forbidden_roots(), block_forbidden=True)
    with _wall_timer(int(config["wall_time_budget_seconds"])), audit:
        bundle = _load_bundle(config, audit)
        baseline_held = _pair_metrics(bundle, bundle.held_smoke)
        baseline_broad = _broad_metrics(bundle, bundle.broad_held)
        installation = install_v80(bundle.language.model)
        installation.assert_fp32_finite()
        installation.assert_only_adapters_trainable(bundle.language.model)
        optimizer = torch.optim.AdamW(
            installation.parameters(),
            lr=float(config["optimization"]["learning_rate"]),
            weight_decay=float(config["optimization"]["weight_decay"]),
            foreach=False,
            fused=False,
        )
        trace: list[dict[str, Any]] = []
        for update, units in enumerate(bundle.schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            scale = 1.0 / (2.0 * len(units))
            pair_records = [
                _pair_gradient(bundle, unit, scale=scale) for unit in units
            ]
            retention = _broad_retention_backward(
                bundle, installation, bundle.broad_train[update - 1]
            )
            preclip = float(
                torch.nn.utils.clip_grad_norm_(
                    installation.parameters(),
                    float(config["optimization"]["gradient_clip_l2"]),
                )
                .detach()
                .cpu()
            )
            if not math.isfinite(preclip) or preclip <= 0.0:
                raise FloatingPointError("V80 training gradient is zero or nonfinite")
            optimizer.step()
            installation.assert_only_adapters_trainable(bundle.language.model)
            installation.assert_fp32_finite()
            trace.append(
                {
                    "update": update,
                    "unit_count": len(units),
                    "unit_keys": [[unit.pair_id, unit.question_key] for unit in units],
                    "pair_records": pair_records,
                    "broad_retention": retention,
                    "preclip_gradient_l2": preclip,
                    "adapter_state_sha256": _state_sha256(installation),
                    "memory": _assert_memory(config, phase=f"after_update_{update}"),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "v80_update",
                        "update": update,
                        "updates": 16,
                        "preclip_gradient_l2": preclip,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        candidate_held = _pair_metrics(bundle, bundle.held_smoke)
        candidate_broad = _broad_metrics(bundle, bundle.broad_held)
        retention = _retention_metrics(bundle, installation, bundle.broad_train)
        pair_delta = _pair_delta(baseline_held, candidate_held)
        broad_delta = candidate_broad["mean_answer_nll"] - baseline_broad["mean_answer_nll"]
        gates = config["held_smoke_gates"]
        checks = {
            "mean_wrong_scene_margin_gain": pair_delta["mean_wrong_scene_margin"]
            >= float(gates["mean_wrong_scene_margin_gain_nats_minimum"]),
            "mean_wrong_scene_softplus_delta": pair_delta["mean_wrong_scene_softplus"]
            <= float(gates["mean_wrong_scene_softplus_delta_nats_maximum"]),
            "mean_correct_answer_nll_delta": pair_delta["mean_correct_nll"]
            <= float(gates["mean_correct_answer_nll_delta_nats_maximum"]),
            "additional_positive_margin_sides": pair_delta["positive_margin_sides"]
            >= int(gates["additional_positive_margin_sides_minimum"]),
            "additional_complete_units": pair_delta["complete_units"]
            >= int(gates["additional_complete_units_minimum"]),
            "minimum_change_family_margin_delta": min(pair_delta["family_margin"].values())
            >= float(gates["minimum_change_family_margin_delta_nats"]),
            "broad_held_nll_delta": broad_delta
            <= float(gates["broad_held_mean_answer_nll_delta_nats_maximum"]),
            "retention_mean_kl": retention["mean_kl_nats"]
            <= float(gates["retention_mean_kl_nats_maximum"]),
            "retention_maximum_kl": retention["maximum_kl_nats"]
            <= float(gates["retention_maximum_kl_nats_maximum"]),
            "retention_top1": retention["answer_token_top1_agreement"]
            >= float(gates["retention_answer_token_top1_agreement_minimum"]),
            "checkpoint_absent": not prohibited.exists(),
        }
    audit.assert_clean()
    checks["file_audit_clean"] = True
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "artifact": "gemma4_v80_v75_atlas_attention_reader_bounded_screen_v1",
        "status": "held_smoke_pass_diagnostic_only_not_promoted"
        if passed
        else "held_smoke_fail_terminal_no_checkpoint",
        "passed": passed,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "gradient_smoke_sha256": sha256_file(smoke_path),
        "optimizer_constructed_after_authenticated_smoke": True,
        "optimizer_updates": len(trace),
        "baseline_held": baseline_held,
        "candidate_held": candidate_held,
        "held_delta": pair_delta,
        "baseline_broad_held": baseline_broad,
        "candidate_broad_held": candidate_broad,
        "broad_held_nll_delta": broad_delta,
        "retention": retention,
        "checks": checks,
        "trace": trace,
        "audit": _audit_summary(audit),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_published": False,
        "runtime_promotion_authorized": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "deferred_final_loaded": False,
        "oracle_loaded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument(
        "--phase", choices=("gradient-smoke", "bounded-screen"), required=True
    )
    args = parser.parse_args(argv)
    config = load_v80_config(args.config)
    output_key = "gradient_smoke" if args.phase == "gradient-smoke" else "bounded_screen"
    try:
        report = (
            run_gradient_smoke(args.config)
            if args.phase == "gradient-smoke"
            else run_bounded_screen(args.config)
        )
    except BaseException as error:  # noqa: BLE001 - emit fail-closed terminal evidence
        report = {
            "schema_version": 1,
            "artifact": f"gemma4_v80_v75_atlas_attention_reader_{args.phase}_failure_v1",
            "status": "terminal_failure_no_checkpoint",
            "passed": False,
            "phase": args.phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "optimizer_updates_unknown_or_zero": args.phase == "gradient-smoke",
            "checkpoint_published": False,
            "runtime_promotion_authorized": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "deferred_final_loaded": False,
            "oracle_loaded": False,
        }
    path, digest = atomic_create_json(config["outputs"][output_key], report)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": digest,
                "status": report["status"],
                "passed": report["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CPU_PREFLIGHT_CORRECTION_SHA256",
    "CPU_PREFLIGHT_SHA256",
    "PREREGISTRATION_SHA256",
    "V80Bundle",
    "WallTimeExceededV80",
    "main",
    "run_bounded_screen",
    "run_gradient_smoke",
]
