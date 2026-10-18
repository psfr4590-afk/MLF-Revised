"""Hardware detection and conservative training-profile selection."""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


def _command_output(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _ram_gb(available: bool) -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                value = status.ullAvailPhys if available else status.ullTotalPhys
                return round(value / (1024**3), 2)
        elif hasattr(os, "sysconf"):
            pages = os.sysconf("SC_AVPHYS_PAGES" if available else "SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * size / (1024**3), 2)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _nvidia() -> tuple[str | None, float | None, str | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None, None
    output = _command_output([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if not output:
        return None, None, None
    parts = [p.strip() for p in output.splitlines()[0].split(",")]
    name = parts[0] if parts else None
    try:
        vram = round(float(parts[1]) / 1024, 2) if len(parts) > 1 else None
    except ValueError:
        vram = None
    driver = parts[2] if len(parts) > 2 else None
    return name, vram, driver


@dataclass(frozen=True)
class HardwareProfile:
    platform: str
    cpu_cores: int
    available_ram_gb: float | None
    gpu_count: int
    gpu_memory_gb: float
    gpu_name: str | None
    cuda_available: bool
    total_ram_gb: float | None = None
    cpu_threads: int | None = None
    cpu_name: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    pytorch_version: str | None = None
    os_name: str | None = None
    disk_free_gb: float | None = None
    disk_total_gb: float | None = None

    @property
    def tier(self) -> str:
        if not self.cuda_available or self.gpu_count == 0:
            return "cpu"
        vram = self.gpu_memory_gb
        if vram < 4:
            return "gpu_<4gb"
        if vram < 6:
            return "gpu_4_6gb"
        if vram < 10:
            return "gpu_6_10gb"
        if vram < 20:
            return "gpu_10_20gb"
        return "gpu_20gb_plus"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tier"] = self.tier
        return data


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    model_preset: str
    seq_len: int
    batch_size: int
    grad_accum_steps: int
    eval_batches: int
    eval_every_steps: int
    checkpoint_every_steps: int
    precision: str
    reason: str
    recommended_steps: int | None = None
    estimated_hours: float | None = None
    decision_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["decision_reasons"] = list(self.decision_reasons)
        return data


def profile_hardware(path: str | Path | None = None) -> HardwareProfile:
    gpu_name, gpu_memory_gb, driver = _nvidia()
    cuda_available = False
    cuda_version = None
    pytorch_version = None
    gpu_count = 1 if gpu_name else 0
    try:
        import torch
        pytorch_version = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_available:
            gpu_count = int(torch.cuda.device_count())
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            gpu_memory_gb = round(float(props.total_memory) / (1024**3), 2)
    except Exception:
        pass
    disk = shutil.disk_usage(Path(path or os.getcwd()).resolve())
    return HardwareProfile(
        platform=platform.platform(),
        cpu_cores=os.cpu_count() or 1,
        available_ram_gb=_ram_gb(True),
        gpu_count=gpu_count,
        gpu_memory_gb=float(gpu_memory_gb or 0.0),
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        total_ram_gb=_ram_gb(False),
        cpu_threads=os.cpu_count() or 1,
        cpu_name=platform.processor() or None,
        cuda_version=str(cuda_version) if cuda_version else None,
        driver_version=driver,
        pytorch_version=str(pytorch_version) if pytorch_version else None,
        os_name=platform.system(),
        disk_free_gb=round(disk.free / (1024**3), 2),
        disk_total_gb=round(disk.total / (1024**3), 2),
    )


def estimate_total_tokens(shard_dir: str | Path) -> int:
    root = Path(shard_dir)
    manifest = root / "shards.manifest.json"
    if manifest.is_file():
        import json
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        dtype = str(payload.get("dtype", "uint16"))
        itemsize = 4 if dtype == "uint32" else 2
        files = payload.get("files", [])
        if not isinstance(files, list):
            raise ValueError(f"Invalid shard manifest files field: {manifest}")
        return sum(int(item.get("size", 0)) // itemsize for item in files)
    files = sorted(root.glob("shard_*_*.bin"))
    if not files:
        raise FileNotFoundError(f"No shard files found in {root}")
    return sum(p.stat().st_size // 2 for p in files)


def recommend_training_profile(
    hardware: HardwareProfile,
    total_tokens: int | None = None,
    configured_steps: int = 100_000,
    target_training_hours: float | None = None,
    observed_tokens_per_sec: float | None = None,
    max_seq_len: int | None = None,
) -> TrainingProfile:
    profiles = {
        "cpu": TrainingProfile("cpu-safe", "85M", 256, 1, 32, 2, 100, 250, "fp32", "CPU-only host; smallest preset and short context reduce memory pressure"),
        "gpu_<4gb": TrainingProfile("gpu-sub4gb", "85M", 128, 1, 32, 2, 100, 250, "fp16", "VRAM < 4 GB; prioritize viability over throughput"),
        "gpu_4_6gb": TrainingProfile("gpu-4-6gb", "85M", 512, 1, 32, 4, 200, 500, "fp16", "4–6 GB VRAM; 85M preset with reduced context"),
        "gpu_6_10gb": TrainingProfile("gpu-6-10gb", "117M", 512, 1, 32, 4, 250, 500, "fp16", "6–10 GB VRAM; 117M is viable with reduced context"),
        "gpu_10_20gb": TrainingProfile("gpu-10-20gb", "117M", 1024, 1, 16, 8, 500, 1000, "fp16", "10–20 GB VRAM; full 1K context on the 117M preset"),
        "gpu_20gb_plus": TrainingProfile("gpu-20gb-plus", "360M", 1024, 1, 16, 8, 500, 1000, "fp16", "20+ GB VRAM; 360M remains substantially below workstation-class models"),
    }
    base = profiles[hardware.tier]
    seq_len = min(base.seq_len, int(max_seq_len)) if max_seq_len is not None else base.seq_len
    if seq_len <= 0:
        raise ValueError("max_seq_len must be > 0 when configured")
    reasons = [base.reason]
    if hardware.gpu_memory_gb and hardware.gpu_memory_gb < 4:
        reasons.append("VRAM < 4 GB")
    if seq_len != base.seq_len:
        reasons.append(f"sequence length explicitly capped to {seq_len}")
    steps = max(1, int(configured_steps))
    tokens_per_step = base.batch_size * base.grad_accum_steps * seq_len
    if total_tokens is not None and total_tokens > 0:
        corpus_steps = max(1, math.ceil(total_tokens / tokens_per_step))
        steps = min(steps, corpus_steps)
    hours = None
    if target_training_hours is not None and target_training_hours > 0 and observed_tokens_per_sec and observed_tokens_per_sec > 0:
        steps_for_target = int(target_training_hours * 3600 * observed_tokens_per_sec / tokens_per_step)
        if steps_for_target < 1:
            steps = min(steps, 1)
            reasons.append(
                "target duration is shorter than one training step; "
                "minimum one step retained"
            )
        else:
            steps = min(steps, steps_for_target)
        hours = (steps * tokens_per_step) / observed_tokens_per_sec / 3600
    return TrainingProfile(
        name=base.name, model_preset=base.model_preset, seq_len=seq_len,
        batch_size=base.batch_size, grad_accum_steps=base.grad_accum_steps,
        eval_batches=base.eval_batches, eval_every_steps=base.eval_every_steps,
        checkpoint_every_steps=base.checkpoint_every_steps, precision=base.precision,
        reason="; ".join(reasons),
        recommended_steps=steps, estimated_hours=hours,
        decision_reasons=tuple(reasons),
    )


def capability_candidates(hardware: HardwareProfile) -> tuple[tuple[str, int], ...]:
    """Return an expandable capability-probe matrix.

    These are probe points, not claims about hardware limits. The preflight
    benchmark determines which combinations actually work.
    """
    if not hardware.cuda_available or hardware.gpu_count == 0:
        return (("85M", 256), ("85M", 512))
    return (
        ("85M", 128), ("85M", 256), ("85M", 512), ("85M", 768), ("85M", 1024),
        ("117M", 256), ("117M", 512), ("117M", 768), ("117M", 1024),
        ("360M", 256), ("360M", 512), ("360M", 768), ("360M", 1024),
    )


__all__ = [
    "HardwareProfile", "TrainingProfile", "profile_hardware",
    "estimate_total_tokens", "recommend_training_profile", "capability_candidates",
]
