"""Unified, non-destructive system readiness assessment for Model Lab.

The assessment combines existing doctor, hardware, and configuration checks.
A benchmark is opt-in and is isolated in a subprocess by the CLI so it can be
terminated if it exceeds its wall-clock budget.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MIN_FREE_DISK_GIB = 2.0
LOW_RAM_GIB = 8.0


def _git_identity(root: Path) -> dict[str, str | None]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=3, check=False,
        )
        sha = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        sha = None
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root, capture_output=True, text=True, timeout=3, check=False,
        )
        branch = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        branch = None
    return {"commit": sha or None, "branch": branch or None}


def _available_ram_gib() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().available / (1024 ** 3), 2)
    except Exception:
        return None


def _disk_state(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": "project filesystem",
            "free_gib": round(usage.free / (1024 ** 3), 2),
            "total_gib": round(usage.total / (1024 ** 3), 2),
            "available": True,
        }
    except OSError as exc:
        return {"path": str(path), "free_gib": None, "total_gib": None,
                "available": False, "error": f"{type(exc).__name__}: {exc}"}


def _backend_smoke(torch_state: dict[str, Any]) -> dict[str, Any]:
    """Execute tiny real tensor operations instead of trusting availability flags."""
    result: dict[str, Any] = {
        "cpu": {"status": "UNKNOWN", "passed": None, "detail": "Not tested."},
        "cuda": {"status": "NOT_AVAILABLE", "passed": None, "detail": "PyTorch did not report CUDA available."},
    }
    try:
        torch = importlib.import_module("torch")
        value = torch.ones((1,), device="cpu")
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError("CPU tensor returned a non-finite value")
        result["cpu"] = {"status": "PASS", "passed": True, "detail": "CPU tensor allocation and operation succeeded."}
    except Exception as exc:
        result["cpu"] = {"status": "BLOCKED", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    if torch_state.get("cuda_available"):
        try:
            torch = importlib.import_module("torch")
            value = torch.ones((1,), device="cuda")
            torch.cuda.synchronize()
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError("CUDA tensor returned a non-finite value")
            result["cuda"] = {"status": "PASS", "passed": True, "detail": "CUDA tensor allocation, operation, and synchronization succeeded."}
        except Exception as exc:
            result["cuda"] = {"status": "BLOCKED", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}
    return result


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _check(name: str, status: str, detail: str, *, required: bool = True,
           recommendation: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "required": required,
        "detail": detail,
        "recommendation": recommendation,
    }


def build_readiness_report(root: str | Path, benchmark: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect fresh system facts and return a machine-readable readiness report."""
    import bootstrap
    from pipeline.doctor import run_doctor
    from pipeline.model_sizer import profile_hardware, recommend_training_profile
    from scripts.system_info import _nvidia_state, _torch_state

    root = Path(root).resolve()
    generated_at = datetime.now(timezone.utc).isoformat()
    identity = _git_identity(root)
    available_ram = _available_ram_gib()
    disk = _disk_state(root)
    hardware = profile_hardware(root)
    try:
        host_hardware = bootstrap.detect_hardware().as_dict()
    except Exception as exc:
        host_hardware = {"gpu_name": None, "gpu_vendor": None,
                         "error": f"{type(exc).__name__}: {exc}"}
    try:
        torch_backends = bootstrap.torch_runtime_status()
    except Exception as exc:
        torch_backends = {"error": f"{type(exc).__name__}: {exc}"}
    host_gpu_detected = bool(host_hardware.get("gpu_name")) or bool(hardware.gpu_count)
    detected_gpu_name = host_hardware.get("gpu_name") or hardware.gpu_name or "unknown GPU"
    detected_gpu_vendor = host_hardware.get("gpu_vendor") or ("nvidia" if hardware.gpu_count else "unknown")
    _, doctor_checks = run_doctor(root)

    checks: list[dict[str, Any]] = []
    for item in doctor_checks:
        if item["ok"]:
            status = "PASS"
        elif item["required"]:
            status = "BLOCKED"
        else:
            status = "WARN"
        recommendation = None
        if item["name"] == "Python" and status == "BLOCKED":
            recommendation = "Use a supported Python version (3.11 through 3.14) in the project virtual environment."
        elif item["name"] == "PyTorch" and status == "BLOCKED":
            recommendation = "Run the documented bootstrap/install command, then repeat readiness checks."
        elif item["name"] == "pipeline config validation" and status == "BLOCKED":
            recommendation = "Correct config/pipeline_config.yaml validation errors before continuing."
        elif item["name"] == "output writable" and status == "BLOCKED":
            recommendation = "Choose a writable project output directory and check filesystem permissions."
        checks.append(_check(item["name"], status, item["detail"],
                             required=bool(item["required"]), recommendation=recommendation))

    if available_ram is None:
        checks.append(_check("Available system RAM", "UNKNOWN",
                             "The operating system did not provide a current available-memory measurement.",
                             required=False,
                             recommendation="Install/repair psutil or inspect available memory manually. Do not treat unknown memory as unlimited."))
    elif available_ram < 1.5:
        checks.append(_check("Available system RAM", "BLOCKED",
                             f"Only {available_ram:.2f} GiB of system RAM is currently available.",
                             recommendation="Close memory-heavy applications and repeat readiness checks before training."))
    elif available_ram < 3.0:
        checks.append(_check("Available system RAM", "WARN",
                             f"Only {available_ram:.2f} GiB of system RAM is currently available.",
                             required=False,
                             recommendation="Use the smallest smoke profile and avoid concurrent memory-heavy applications."))
    else:
        checks.append(_check("Available system RAM", "PASS",
                             f"{available_ram:.2f} GiB currently available.", required=False))

    if not disk["available"]:
        checks.append(_check("Project disk space", "UNKNOWN", disk.get("error", "Disk usage unavailable"),
                             required=False, recommendation="Check disk capacity and permissions manually."))
    elif disk["free_gib"] < MIN_FREE_DISK_GIB:
        checks.append(_check("Project disk space", "BLOCKED",
                             f"Only {disk['free_gib']:.2f} GiB free on the project filesystem.",
                             recommendation=f"Free at least {MIN_FREE_DISK_GIB:.1f} GiB before running the pipeline. This is a minimum guard, not a full-dataset estimate."))
    elif disk["free_gib"] < 10.0:
        checks.append(_check("Project disk space", "WARN",
                             f"{disk['free_gib']:.2f} GiB free. Larger datasets and checkpoints may require substantially more.",
                             required=False,
                             recommendation="Estimate dataset, shard, checkpoint, and export sizes before a long run."))
    else:
        checks.append(_check("Project disk space", "PASS",
                             f"{disk['free_gib']:.2f} GiB free.", required=False))

    try:
        torch_state = _torch_state()
    except Exception as exc:
        torch_state = {
            "installed": None,
            "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        nvidia_state = _nvidia_state()
    except Exception as exc:
        nvidia_state = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    backend_smoke = _backend_smoke(torch_state)
    checks.append(_check(
        "PyTorch CPU tensor smoke",
        "PASS" if backend_smoke["cpu"]["passed"] is True else "BLOCKED",
        backend_smoke["cpu"]["detail"],
        required=True,
        recommendation=None if backend_smoke["cpu"]["passed"] is True else "Reinstall or repair PyTorch in the project environment, then rerun readiness.",
    ))
    cuda_available = bool(torch_state.get("cuda_available")) and backend_smoke["cuda"]["passed"] is True
    if cuda_available:
        checks.append(_check("CUDA runtime", "PASS",
                             f"PyTorch {torch_state.get('version')} passed a CUDA tensor operation and synchronization on {torch_state.get('device_count', 0)} device(s).",
                             required=False))
    elif hardware.gpu_count and torch_state.get("cuda_available"):
        checks.append(_check("CUDA runtime", "WARN",
                             "PyTorch reported CUDA available, but the real CUDA tensor smoke failed; CPU fallback may still work. " + backend_smoke["cuda"]["detail"],
                             required=False,
                             recommendation="Verify the installed PyTorch build, driver compatibility, and GPU memory availability."))
    elif host_gpu_detected:
        checks.append(_check("CUDA runtime", "WARN",
                             f"GPU detected ({detected_gpu_name}, vendor={detected_gpu_vendor}), but a CUDA training operation was not verified. The current preflight benchmark uses CUDA or CPU; non-CUDA acceleration is not proven by this check.",
                             required=False,
                             recommendation="Use CPU fallback unless a supported PyTorch backend and the training path are independently verified for this GPU."))
    else:
        checks.append(_check("CUDA runtime", "PASS",
                             "No GPU was detected; CPU is the expected execution path on this host.",
                             required=False))

    profile = None
    profile_error = None
    try:
        profile = recommend_training_profile(hardware).to_dict()
        # The existing tier-based defaults do not account for host RAM. Constrain
        # the recommendation when memory is low or cannot be measured.
        if available_ram is None or available_ram < 3.0:
            profile["model_preset"] = "85M"
            profile["seq_len"] = min(int(profile.get("seq_len", 128)), 128)
            profile["batch_size"] = 1
            profile["precision"] = "fp32" if not cuda_available else profile.get("precision", "fp16")
            profile["decision_reasons"] = list(profile.get("decision_reasons", [])) + [
                "Conservative override: available system RAM is unknown or below 3 GiB."
            ]
            profile["reason"] = "; ".join(profile["decision_reasons"])
        elif ((hardware.total_ram_gb is not None and hardware.total_ram_gb <= LOW_RAM_GIB)
              or (available_ram is not None and available_ram < 3.0)):
            profile["model_preset"] = "85M"
            profile["seq_len"] = min(int(profile.get("seq_len", 128)), 256)
            profile["batch_size"] = 1
            profile["decision_reasons"] = list(profile.get("decision_reasons", [])) + [
                "Conservative override: total system RAM is 8 GiB or less, or less than 3 GiB is currently available."
            ]
            profile["reason"] = "; ".join(profile["decision_reasons"])
    except Exception as exc:
        profile_error = f"{type(exc).__name__}: {exc}"
        checks.append(_check("Training profile", "UNKNOWN", profile_error, required=False,
                             recommendation="Inspect hardware detection and model_sizer profile selection."))

    if benchmark is None:
        benchmark_result: dict[str, Any] = {
            "status": "NOT_RUN",
            "detail": "No training benchmark was requested. Hardware detection alone does not prove model-training viability.",
        }
        checks.append(_check(
            "Training capability benchmark", "UNKNOWN",
            "No empirical model-training benchmark has been run. This does not block dataset preparation, but training viability remains unverified.",
            required=False,
            recommendation="After verified training shards exist, run python scripts/system_readiness.py --benchmark to test the configured model in an isolated, time-limited process.",
        ))
    elif benchmark.get("viable") is True:
        benchmark_result = {"status": "PASS", "detail": "Bounded training benchmark completed.", "result": benchmark}
        checks.append(_check("Training capability benchmark", "PASS",
                             f"{benchmark.get('tokens_per_sec')} tokens/sec measured; {benchmark.get('benchmark_seconds')} seconds elapsed.",
                             required=False))
    else:
        benchmark_result = {"status": "BLOCKED", "detail": benchmark.get("failure") or "Benchmark did not complete.", "result": benchmark}
        checks.append(_check("Training capability benchmark", "BLOCKED",
                             benchmark_result["detail"], required=False,
                             recommendation="Review the benchmark error, lower model/context settings, or verify the training shards and runtime."))

    blocking = [item for item in checks if item["status"] == "BLOCKED" and item["required"]]
    warnings = [item for item in checks if item["status"] in ("WARN", "UNKNOWN") or
                (item["status"] == "BLOCKED" and not item["required"])]
    if blocking:
        overall = "BLOCKED"
    elif warnings:
        overall = "READY_WITH_LIMITS"
    else:
        overall = "READY"

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "project": "MLF-Revised",
        "identity": identity,
        "runtime": {
            "python_version": sys.version.split()[0],
            "python_executable_name": Path(sys.executable).name,
            "platform": sys.platform,
            "project_root_name": root.name,
        },
        "hardware": hardware.to_dict(),
        "host_hardware": host_hardware,
        "memory": {"available_ram_gib": available_ram},
        "disk": disk,
        "torch": torch_state,
        "torch_backends": torch_backends,
        "backend_smoke": backend_smoke,
        "configuration": {"pipeline_config_sha256": _sha256_file(root / "config" / "pipeline_config.yaml")},
        "nvidia": nvidia_state,
        "checks": checks,
        "training_profile": profile,
        "training_profile_error": profile_error,
        "benchmark": benchmark_result,
        "summary": {
            "status": overall,
            "passed": sum(item["status"] == "PASS" for item in checks),
            "warnings": len(warnings),
            "blockers": len(blocking),
            "required_checks_passed": not blocking,
        },
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    return report


def write_readiness_report(report: dict[str, Any], output: str | Path) -> Path:
    """Atomically persist a report without leaving a partial JSON file."""
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


__all__ = ["build_readiness_report", "write_readiness_report"]
