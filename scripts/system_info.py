#!/usr/bin/env python3
"""Generate a host-specific installation and hardware verification report.

This module records measured host facts. It does not infer CUDA availability from
NVIDIA hardware and it does not persist usernames, hostnames, or other identity
metadata. Generated reports are intentionally machine-specific and belong
outside source control.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bootstrap  # noqa: E402

DEFAULT_OUTPUT = ROOT / "runtime" / "system_info.json"


def _command(command: list[str], timeout: float = 10.0) -> tuple[int | None, str]:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, type(exc).__name__
    return result.returncode, (result.stdout or result.stderr).strip()


def _pip_check() -> dict[str, Any]:
    code, output = _command([sys.executable, "-m", "pip", "check"], timeout=30.0)
    return {"passed": code == 0, "exit_code": code, "output": output}


def _nvidia_state() -> dict[str, Any]:
    code, output = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if code is None:
        return {"available": False, "error": output}
    if code != 0 or not output:
        return {"available": False, "error": output or "nvidia-smi failed"}
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            vram_mib = float(parts[2])
        except ValueError:
            vram_mib = None
        rows.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "vram_gib": round(vram_mib / 1024, 2) if vram_mib is not None else None,
            }
        )
    return {"available": bool(rows), "gpus": rows}


def _torch_state() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "cuda_available": False, "error": type(exc).__name__}

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "device_count": len(devices),
        "devices": devices,
    }


def build_report(torch_channel: str = "cpu") -> dict[str, Any]:
    """Collect measured host, dependency, and Torch runtime state."""
    profile = bootstrap.detect_hardware()
    torch_state = _torch_state()
    nvidia_state = _nvidia_state()
    pip_check = _pip_check()
    constrained = (
        profile.ram_gib is not None and profile.ram_gib <= 8.0
    ) or (
        profile.gpu_vram_gib is not None and profile.gpu_vram_gib <= 4.0
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "ML-Framework",
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": Path(sys.executable).name,
            "machine": platform.machine(),
        },
        "hardware": profile.as_dict(),
        "nvidia": nvidia_state,
        "torch": torch_state,
        "installation": {
            "selected_torch_channel": torch_channel,
            "pip_check": pip_check,
            "verified": bool(torch_state.get("installed")) and pip_check["passed"],
        },
        "assessment": {
            "constrained_machine": constrained,
            "cuda_claim": "runtime_verified" if torch_state.get("cuda_available") else "not_runtime_verified",
        },
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate measured ML-Framework host system information")
    parser.add_argument("--torch-channel", choices=("cpu", "default"), default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.torch_channel)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["installation"]["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
