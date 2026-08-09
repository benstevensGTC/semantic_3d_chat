from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]


def command_version(command: str, *args: str) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    result = subprocess.run(
        [executable, *args], capture_output=True, text=True, check=False, timeout=30
    )
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else executable


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--skip-mps-smoke",
        action="store_true",
        help="Report MPS availability without allocating an MPS tensor.",
    )
    result.add_argument(
        "--output",
        default="reports/metrics/machine_report.json",
        help="Destination for the JSON machine report, relative to the project root.",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    disk = shutil.disk_usage(ROOT)
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    mps_smoke: bool | None = None if args.skip_mps_smoke else False
    mps_error = None
    if mps_available and not args.skip_mps_smoke:
        try:
            value = (torch.ones(4, device="mps") @ torch.ones(4, device="mps")).item()
            mps_smoke = value == 4.0
        except Exception as exc:  # noqa: BLE001 - record any backend diagnostic failure
            mps_error = repr(exc)
    report = {
        "architecture": platform.machine(),
        "processor": platform.processor() or "Apple Silicon",
        "macos_version": platform.mac_ver()[0],
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "blender": command_version("blender", "--version"),
        "homebrew": command_version("brew", "--version"),
        "uv": command_version("uv", "--version"),
        "torch_version": torch.__version__,
        "torch_mps_built": mps_built,
        "torch_mps_available": mps_available,
        "torch_mps_smoke": mps_smoke,
        "torch_mps_smoke_skipped": args.skip_mps_smoke,
        "torch_mps_error": mps_error,
    }
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if platform.machine() != "arm64":
        print("warning: non-Apple-Silicon architecture; CPU fallback will be used", file=sys.stderr)
    if not report["blender"]:
        raise SystemExit("Blender is missing")


if __name__ == "__main__":
    main()
