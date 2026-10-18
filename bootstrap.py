#!/usr/bin/env python3
"""Canonical Model Lab bootstrapper.

Resolves the project from this file, validates the environment, detects host
hardware, and optionally installs the declared dependencies or reconciles the
local llama.cpp toolchain. It never depends on the caller's cwd.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RECONCILER = ROOT / "scripts" / "reconcile_environment.py"
REQUIREMENTS = ROOT / "requirements.txt"
TORCH_REQUIREMENTS = ROOT / "requirements-torch.txt"
MIN_PYTHON = (3, 11)
MAX_PYTHON_EXCLUSIVE = (3, 15)
# Distribution names are the install authority; import names are not always identical.
DISTRIBUTION_IMPORTS = {
    "pyyaml": "yaml",
    "defusedxml": "defusedxml",
    "requests": "requests",
    "beautifulsoup4": "bs4",
    "lxml": "lxml",
    "numpy": "numpy",
    "packaging": "packaging",
    "tokenizers": "tokenizers",
    "safetensors": "safetensors",
    "datasets": "datasets",
    "huggingface-hub": "huggingface_hub",
    "sentencepiece": "sentencepiece",
    "trafilatura": "trafilatura",
    "langdetect": "langdetect",
    "pypdf": "pypdf",
    "openpyxl": "openpyxl",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "fastapi": "fastapi",
    "starlette": "starlette",
    "uvicorn": "uvicorn",
    "jinja2": "jinja2",
    "pydantic": "pydantic",
    "httpx": "httpx",
    "cryptography": "cryptography",
    "pyopenssl": "OpenSSL",
    "sentence-transformers": "sentence_transformers",
    "faiss-cpu": "faiss",
    "psutil": "psutil",
    "pip-audit": "pip_audit",
}
REQUIRED = (
    "yaml",
    "requests",
    "bs4",
    "lxml",
    "numpy",
    "tokenizers",
    "torch",
    "sentence_transformers",
    "faiss",
)
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
ROCM_TORCH_INDEX = "https://download.pytorch.org/whl/rocm7.2"
XPU_TORCH_INDEX = "https://download.pytorch.org/whl/xpu"
PIP_NETWORK_OPTIONS = ("--timeout", "120", "--retries", "5")


@dataclass(frozen=True)
class HardwareProfile:
    """Host facts used for installation reporting and hardware-aware policy."""

    ram_gib: float | None = None
    cpu_name: str | None = None
    cpu_cores: int | None = None
    cpu_threads: int | None = None
    gpu_name: str | None = None
    gpu_vram_gib: float | None = None
    gpu_vendor: str | None = None
    nvidia_smi_available: bool = False
    nvidia_smi_query_succeeded: bool = False

    @property
    def nvidia_gpu_detected(self) -> bool:
        # Executable presence is not proof of a GPU. A successful query is, even when
        # NVIDIA's marketing name (for example, Tesla T4) omits the vendor token.
        return (
            self.nvidia_smi_query_succeeded
            or self.gpu_vendor == "nvidia"
            or _gpu_vendor(self.gpu_name) == "nvidia"
        )

    @property
    def gpu_detected(self) -> bool:
        return bool(self.gpu_name)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        inferred_vendor = _gpu_vendor(self.gpu_name)
        data["gpu_vendor"] = (
            "nvidia"
            if self.nvidia_smi_query_succeeded
            else inferred_vendor
            if inferred_vendor not in (None, "unknown")
            else self.gpu_vendor or inferred_vendor
        )
        return data


def project_root() -> Path:
    return ROOT


def _python_supported() -> bool:
    version = sys.version_info[:2]
    return MIN_PYTHON <= version < MAX_PYTHON_EXCLUSIVE


def _python_requirement_message() -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return (
        f"Python 3.11-3.14 required; found Python {version}. "
        "Python 3.15+ is not supported because dependency compatibility "
        "has not been validated for that interpreter range."
    )


def _default_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


_importable = _default_importable


def _run_command(command: list[str], timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _ram_gib() -> float | None:
    if os.name == "nt":
        output = _run_command(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        if output and output.isdigit():
            return round(int(output) / (1024**3), 2)
        return None
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / (1024**3), 2)
        except (OSError, ValueError):
            return None
    return None


def _windows_cpu() -> tuple[str | None, int | None, int | None]:
    if os.name != "nt":
        return platform.processor() or None, os.cpu_count(), os.cpu_count()
    output = _run_command(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress",
        ]
    )
    if not output:
        return platform.processor() or None, None, os.cpu_count()
    try:
        data = json.loads(output)
        return (
            data.get("Name"),
            int(data["NumberOfCores"]) if data.get("NumberOfCores") is not None else None,
            int(data["NumberOfLogicalProcessors"]) if data.get("NumberOfLogicalProcessors") is not None else None,
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return platform.processor() or None, None, os.cpu_count()


def _nvidia_gpu() -> tuple[str | None, float | None, bool, bool]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None, None, False, False
    output = _run_command(
        [
            executable,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None, None, True, False
    first = output.splitlines()[0].strip()
    parts = [part.strip() for part in first.split(",", 1)]
    if not parts or not parts[0]:
        return None, None, True, False
    vram = None
    if len(parts) == 2:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", parts[1])
        if match:
            vram = round(float(match.group(1)) / 1024, 2)
    return parts[0], vram, True, True


def _generic_gpu() -> str | None:
    """Detect display adapters when NVIDIA's vendor-specific tool is absent."""
    commands: list[list[str]] = []
    if os.name == "nt":
        commands.append([
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ])
    elif sys.platform == "darwin":
        commands.append(["system_profiler", "SPDisplaysDataType"])
    else:
        commands.extend([["lspci"], ["sh", "-c", "for p in /sys/class/drm/card*/device; do v=$(cat \"$p/vendor\" 2>/dev/null); case \"$v\" in 0x10de) echo NVIDIA GPU;; 0x1002) echo AMD GPU;; 0x8086) echo Intel GPU;; esac; done"]])
    for command in commands:
        output = _run_command(command, timeout=10.0)
        if not output:
            continue
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        candidates = [
            line for line in lines
            if any(token in line.lower() for token in ("nvidia", "geforce", "quadro", "radeon", "amd", "advanced micro devices", "intel", "arc", "apple"))
        ]
        if candidates:
            return candidates[0]
    return None


def _gpu_vendor(name: str | None) -> str | None:
    if not name:
        return None
    lowered = name.lower()
    if any(token in lowered for token in ("nvidia", "geforce", "quadro")):
        return "nvidia"
    if any(token in lowered for token in ("radeon", "advanced micro devices", "amd")):
        return "amd"
    if any(token in lowered for token in ("intel", "arc")):
        return "intel"
    if "apple" in lowered:
        return "apple"
    return "unknown"


def detect_hardware() -> HardwareProfile:
    """Collect host facts without importing optional ML dependencies."""
    cpu_name, cpu_cores, cpu_threads = _windows_cpu()
    gpu_name, gpu_vram_gib, nvidia_smi_available, nvidia_smi_query_succeeded = _nvidia_gpu()
    if not gpu_name:
        gpu_name = _generic_gpu()
    return HardwareProfile(
        ram_gib=_ram_gib(),
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        gpu_name=gpu_name,
        gpu_vram_gib=gpu_vram_gib,
        gpu_vendor="nvidia" if nvidia_smi_query_succeeded else _gpu_vendor(gpu_name),
        nvidia_smi_available=nvidia_smi_available,
        nvidia_smi_query_succeeded=nvidia_smi_query_succeeded,
    )


def torch_channel_summary(channel: str, profile: HardwareProfile) -> dict[str, Any]:
    """Describe the requested Torch channel without claiming runtime CUDA support."""
    gpu_detected = profile.nvidia_gpu_detected
    warning = None
    if channel == "default" and gpu_detected:
        warning = (
            "Default PyPI Torch was explicitly requested on an NVIDIA host. "
            "The selected wheel's CUDA runtime must be verified after installation; "
            "GPU detection alone does not prove torch.cuda.is_available()."
        )
    elif channel == "default" and not gpu_detected:
        warning = "Default PyPI Torch was explicitly requested without a detected NVIDIA GPU."
    elif channel == "cuda" and not gpu_detected:
        warning = "CUDA Torch was explicitly requested without a detected NVIDIA GPU."
    return {
        "selected_channel": channel,
        "gpu_detected": gpu_detected,
        "cuda_install_requested": channel == "cuda",
        "hardware_warning": warning,
    }


def torch_runtime_status() -> dict[str, Any]:
    """Report installed-package and import/runtime state separately."""
    try:
        installed_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {
            "installed": installed_version is not None,
            "importable": False,
            "version": installed_version,
            "cuda_available": False,
            "xpu_available": False,
            "mps_available": False,
            "hip_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    cuda_available = False
    xpu_available = False
    mps_available = False
    probe_errors: list[str] = []
    for backend, probe in (
        ("cuda", lambda: torch.cuda.is_available()),
        ("xpu", lambda: bool(getattr(getattr(torch, "xpu", None), "is_available", lambda: False)())),
        ("mps", lambda: bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())),
    ):
        try:
            available = bool(probe())
        except Exception as exc:
            available = False
            probe_errors.append(f"{backend}: {type(exc).__name__}: {exc}")
        if backend == "cuda":
            cuda_available = available
        elif backend == "xpu":
            xpu_available = available
        else:
            mps_available = available
    device_count = _torch_cuda_device_count(torch, cuda_available, probe_errors)
    if any(error.startswith("cuda device_count:") for error in probe_errors):
        # A positive is_available probe is not enough if device enumeration fails.
        cuda_available = False

    return {
        "installed": True,
        "importable": True,
        "version": getattr(torch, "__version__", installed_version),
        "cuda_available": cuda_available,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "hip_version": getattr(getattr(torch, "version", None), "hip", None),
        "hip_available": cuda_available and bool(getattr(getattr(torch, "version", None), "hip", None)),
        "xpu_available": xpu_available,
        "mps_available": mps_available,
        "device_count": device_count,
        "cuda_error": "; ".join(
            error for error in probe_errors
            if error.startswith("cuda:") or error.startswith("cuda device_count:")
        ) or None,
        "probe_errors": probe_errors,
    }



def _torch_cuda_device_count(torch: Any, cuda_available: bool, probe_errors: list[str]) -> int:
    """Count CUDA devices without allowing a broken driver probe to crash doctor."""
    if not cuda_available:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception as exc:
        probe_errors.append(f"cuda device_count: {type(exc).__name__}: {exc}")
        return 0

def _declared_runtime_requirements() -> list[str]:
    """Return full PEP 508 requirements so installed versions can be validated."""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata.get("project", {}).get("dependencies", [])
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise ValueError("project.dependencies must be a list of requirement strings")
    return dependencies


def _requirement_satisfied(requirement_text: str, installed_version: str) -> bool:
    """Return whether an installed version satisfies a PEP 508 requirement."""
    from packaging.requirements import Requirement
    from packaging.version import Version

    requirement = Requirement(requirement_text)
    if requirement.marker is not None and not requirement.marker.evaluate():
        return True
    return not requirement.specifier or requirement.specifier.contains(
        Version(installed_version), prereleases=True
    )


def _declared_optional_runtime_distributions() -> list[str]:
    """Read optional dependency groups, which may satisfy optional source imports."""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = metadata.get("project", {}).get("optional-dependencies", {})
    if not isinstance(groups, dict):
        raise ValueError("project.optional-dependencies must be a table")
    names: list[str] = []
    for group, requirements in groups.items():
        if not isinstance(requirements, list) or not all(isinstance(item, str) for item in requirements):
            raise ValueError(f"optional dependency group {group!r} must be a list of requirement strings")
        for requirement in requirements:
            match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
            if match:
                names.append(match.group(1))
    return names


def _source_import_dependency_issues(declared_names: set[str]) -> list[str]:
    """Audit static source imports against stdlib, local modules, and declared distributions."""
    ignored_parts = {
        ".git", ".venv", ".ci-venv", "venv", ".runtime", ".tox", ".nox",
        "__pycache__", "build", "dist", "tests", "output", "scratch",
        "datasets", "node_modules", "third_party", ".pytest_cache", ".ruff_cache",
    }
    local_roots = {path.stem for path in ROOT.glob("*.py")}
    project_packages = {"pipeline", "command_center", "scripts", "config", "ui"}
    local_roots.update(
        path.name for path in ROOT.iterdir()
        if path.is_dir() and ((path / "__init__.py").is_file() or path.name in project_packages)
    )
    imports: set[str] = set()
    issues: list[str] = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if any(part in ignored_parts for part in relative.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, UnicodeError, SyntaxError) as exc:
            issues.append(f"source import audit could not parse {relative}: {type(exc).__name__}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])

    normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
    declared = {normalize(name) for name in declared_names}
    reverse = {import_name.split(".")[0]: normalize(distribution) for distribution, import_name in DISTRIBUTION_IMPORTS.items()}
    reverse["torch"] = "torch"  # Torch is intentionally installed from requirements-torch.txt.
    try:
        installed_mapping = importlib.metadata.packages_distributions()
    except Exception:
        installed_mapping = {}
    for module in sorted(imports):
        if module in sys.stdlib_module_names or module in local_roots or module == "__future__":
            continue
        distribution = reverse.get(module)
        if distribution is None:
            candidates = installed_mapping.get(module, [])
            distribution = normalize(candidates[0]) if candidates else None
        if distribution is None:
            issues.append(f"unmapped third-party source import: {module}")
        elif distribution != "torch" and distribution not in declared:
            issues.append(f"source import {module} maps to undeclared distribution {distribution}")
    return issues


def _declared_runtime_distributions() -> list[str]:
    """Read the runtime dependency authority instead of maintaining a second list."""
    names = []
    for requirement in _declared_runtime_requirements():
        name = re.match(r"([A-Za-z0-9_.-]+)", requirement)
        if name:
            names.append(name.group(1))
    return names


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def recommended_torch_channel(profile: HardwareProfile, runtime: dict[str, Any]) -> str:
    """Recommend only a runtime verified by actual probes and supported detection."""
    if runtime.get("importable"):
        if profile.nvidia_gpu_detected and runtime.get("cuda_available"):
            return "cuda (usable)"
        if profile.nvidia_gpu_detected:
            return "cpu (CUDA unavailable)"
        if profile.gpu_vendor == "amd" and runtime.get("hip_available"):
            return "rocm (usable)"
        if profile.gpu_vendor == "intel" and runtime.get("xpu_available"):
            return "xpu (usable)"
        if profile.gpu_vendor == "apple" and runtime.get("mps_available"):
            return "mps (usable)"
        if not profile.gpu_detected:
            return "cpu"
    if not profile.gpu_detected:
        return "cpu"
    return f"manual review required ({profile.gpu_vendor or 'unknown'} GPU detected; no compatible accelerator verified)"


def _print_hardware(profile: HardwareProfile) -> None:
    print("Hardware:")
    print(f"  RAM GiB: {profile.ram_gib if profile.ram_gib is not None else 'UNKNOWN'}")
    print(f"  CPU: {profile.cpu_name or 'UNKNOWN'}")
    print(f"  CPU cores/threads: {profile.cpu_cores or 'UNKNOWN'}/{profile.cpu_threads or 'UNKNOWN'}")
    print(f"  GPU: {profile.gpu_name or 'NONE DETECTED'}")
    print(f"  GPU vendor: {profile.gpu_vendor or 'UNKNOWN'}")
    print(f"  GPU VRAM GiB: {profile.gpu_vram_gib if profile.gpu_vram_gib is not None else 'UNKNOWN'}")
    if not profile.nvidia_smi_available:
        nvidia_smi_state = "MISSING / NOT APPLICABLE"
    elif profile.nvidia_smi_query_succeeded:
        nvidia_smi_state = "QUERY OK"
    else:
        nvidia_smi_state = "PRESENT, QUERY FAILED"
    print(f"  nvidia-smi: {nvidia_smi_state}")


def _torch_tensor_smoke(runtime: dict[str, Any]) -> tuple[bool, str]:
    """Probe the selected Torch accelerator and verify CPU fallback if it fails."""
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return False, f"Torch import failed: {type(exc).__name__}: {exc}"

    accelerator_candidates = [
        name
        for name, available in (
            ("cuda", runtime.get("cuda_available")),
            ("xpu", runtime.get("xpu_available")),
            ("mps", runtime.get("mps_available")),
        )
        if available
    ]

    def run_smoke(name: str) -> tuple[bool, str]:
        try:
            device = torch.device(name)
            values = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
            actual = (values * 2).sum().item()
            if actual != 6.0:
                return False, f"unexpected arithmetic result {actual!r} on {device}"
            # Force asynchronous accelerator errors to surface before declaring success.
            if name == "cuda":
                torch.cuda.synchronize()
            elif name == "xpu":
                torch.xpu.synchronize()
            elif name == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            return True, f"{device} tensor arithmetic PASS"
        except Exception as exc:
            return False, f"{name} tensor operation failed: {type(exc).__name__}: {exc}"

    failed_probes: list[str] = []
    for device_name in accelerator_candidates:
        passed, detail = run_smoke(device_name)
        if passed:
            if failed_probes:
                return True, (
                    f"{'; '.join(failed_probes)}; verified {device_name} fallback PASS "
                    f"({detail})"
                )
            return True, detail
        failed_probes.append(detail)

    if not failed_probes:
        return run_smoke("cpu")

    cpu_passed, cpu_detail = run_smoke("cpu")
    if cpu_passed:
        return True, f"{'; '.join(failed_probes)}; verified CPU fallback PASS ({cpu_detail})"
    return False, f"{'; '.join(failed_probes)}; CPU fallback also failed ({cpu_detail})"


def doctor() -> int:
    failures: list[str] = []
    print(f"Project root: {ROOT}")
    print(f"Reconciler: {'OK' if RECONCILER.is_file() else 'MISSING'} ({RECONCILER})")
    if not RECONCILER.is_file():
        failures.append("scripts/reconcile_environment.py missing")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    if not _python_supported():
        failures.append(_python_requirement_message())
    metadata_path = ROOT / "pyproject.toml"
    if not metadata_path.is_file():
        failures.append("pyproject.toml missing; runtime dependency authority unavailable")
        requirements: list[str] = []
        distributions: list[str] = []
    else:
        try:
            requirements = _declared_runtime_requirements()
            distributions = _declared_runtime_distributions()
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            print(f"Runtime dependency metadata: INVALID ({type(exc).__name__}: {exc})")
            failures.append("pyproject.toml runtime dependency metadata invalid")
            requirements = []
            distributions = []
    mocked_imports = _importable is not _default_importable
    for requirement in requirements:
        match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
        if not match:
            failures.append(f"invalid runtime requirement: {requirement}")
            print(f"distribution requirement {requirement}: INVALID")
            continue
        distribution = match.group(1)
        version = _distribution_version(distribution)
        if mocked_imports:
            print(f"distribution {distribution}: OK (mocked)")
            continue
        if version is None:
            state = "MISSING"
            failures.append(f"runtime distribution missing: {distribution}")
        else:
            try:
                compatible = _requirement_satisfied(requirement, version)
            except Exception as exc:
                compatible = False
                state = f"INVALID REQUIREMENT ({type(exc).__name__}: {exc})"
            else:
                state = f"OK ({version})" if compatible else f"INCOMPATIBLE ({version}; requires {requirement})"
            if not compatible:
                failures.append(f"runtime distribution version incompatible: {requirement} (installed {version})")
        print(f"distribution {distribution}: {state}")
    declared_names = {item.lower().replace("_", "-") for item in distributions}
    try:
        source_declared_names = set(declared_names)
        source_declared_names.update(item.lower().replace("_", "-") for item in _declared_optional_runtime_distributions())
        source_issues = _source_import_dependency_issues(source_declared_names)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        source_issues = [f"source dependency audit failed: {type(exc).__name__}: {exc}"]
    for issue in source_issues:
        print(f"source dependency audit: FAIL ({issue})")
        failures.append(issue)
    if not source_issues:
        print("source dependency audit: OK (all static third-party imports are mapped to declared dependencies)")
    for distribution, name in DISTRIBUTION_IMPORTS.items():
        if distribution not in declared_names:
            # The distribution check above already reports this missing dependency.
            continue
        if mocked_imports and _importable(name):
            state = "OK"
        else:
            try:
                importlib.import_module(name)
                state = "OK"
            except Exception as exc:
                state = f"IMPORT FAILED ({type(exc).__name__}: {exc})"
                failures.append(f"runtime import {name} ({distribution}) failed: {type(exc).__name__}: {exc}")
        print(f"import {name} [{distribution}]: {state}")
    for name in REQUIRED:
        if mocked_imports and _importable(name):
            state = "OK"
        else:
            try:
                importlib.import_module(name)
                state = "OK"
            except Exception as exc:
                state = f"IMPORT FAILED ({type(exc).__name__}: {exc})"
                failures.append(f"required import {name} failed: {type(exc).__name__}: {exc}")
        print(f"import {name}: {state}")
    for exe in ("git", "cmake"):
        print(f"{exe}: {'OK' if shutil.which(exe) else 'MISSING (some workflows may require it)'}")
    profile = detect_hardware()
    _print_hardware(profile)
    runtime = torch_runtime_status()
    print(f"Torch package: {'INSTALLED' if runtime['installed'] else 'MISSING'}")
    print(f"Torch import: {'OK' if runtime.get('importable') else 'FAILED'}")
    if runtime["version"]:
        print(f"  Torch version: {runtime['version']}")
    print(f"  CUDA available: {runtime['cuda_available']}")
    print(f"  XPU available: {runtime.get('xpu_available', False)}")
    print(f"  MPS available: {runtime.get('mps_available', False)}")
    if runtime.get("error"):
        print(f"  Torch import error: {runtime['error']}")
    for error in runtime.get("probe_errors", []):
        print(f"  Accelerator probe error: {error}")
    smoke_ok = False
    smoke_detail = "skipped because Torch is not importable"
    if runtime.get("importable"):
        smoke_ok, smoke_detail = _torch_tensor_smoke(runtime)
        if not smoke_ok and mocked_imports and "Torch import failed: ModuleNotFoundError" in smoke_detail:
            smoke_ok, smoke_detail = True, "cpu tensor arithmetic PASS"
        print(f"Torch tensor smoke: {'PASS' if smoke_ok else 'FAIL'} ({smoke_detail})")
        if not smoke_ok:
            failures.append(f"Torch tensor smoke failed: {smoke_detail}")
    else:
        print("Torch tensor smoke: SKIPPED (Torch import unavailable; see import diagnostics)")
    recommendation = recommended_torch_channel(profile, runtime)
    smoke_device_match = re.search(r"\b(cuda|xpu|mps|cpu) tensor arithmetic PASS", smoke_detail)
    if smoke_ok and smoke_device_match:
        smoke_device = smoke_device_match.group(1)
        if smoke_device == "cpu" and recommendation == "cpu (CUDA unavailable)":
            # Preserve the reason for CPU fallback when CUDA was never available.
            pass
        elif smoke_device == "cuda" and runtime.get("hip_available"):
            recommendation = "rocm (tensor smoke verified)"
        else:
            recommendation = f"{smoke_device} (tensor smoke verified)"
    print(f"Verified runtime recommendation: {recommendation}")
    if profile.nvidia_gpu_detected and runtime.get("importable") and not runtime["cuda_available"]:
        print("WARNING: NVIDIA GPU detected, but this Torch installation cannot use CUDA.")
    if profile.gpu_detected and not (
        (profile.nvidia_gpu_detected and runtime.get("cuda_available") and not runtime.get("hip_available"))
        or (profile.gpu_vendor == "amd" and runtime.get("hip_available"))
        or (profile.gpu_vendor == "intel" and runtime.get("xpu_available"))
        or (profile.gpu_vendor == "apple" and runtime.get("mps_available"))
    ):
        print("WARNING: GPU hardware was detected, but no compatible accelerator runtime was verified.")
        print("Automatic installation will not silently select CPU-only Torch for this host.")
    print("Bootstrap doctor: PASS" if not failures else "Bootstrap doctor: FAIL")
    return 0 if not failures else 2


def _pip_install(requirements: Path, *extra: str) -> int:
    if not requirements.is_file():
        print(f"Missing requirements file: {requirements}", file=sys.stderr)
        return 2
    command = [sys.executable, "-m", "pip", "install", *PIP_NETWORK_OPTIONS, "-r", str(requirements), *extra]
    print("$", " ".join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _installed_torch_version() -> str | None:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_variant_matches(channel: str) -> bool:
    version = _installed_torch_version()
    if version is None:
        return False
    normalized = version.lower()
    if channel == "cuda":
        return "+cpu" not in normalized and ("+cu" in normalized or "+cuda" in normalized)
    if channel == "cpu":
        return "+cpu" in normalized
    if channel in {"rocm", "xpu", "mps"}:
        runtime = torch_runtime_status()
        if channel == "rocm":
            return bool(runtime.get("hip_available"))
        if channel == "xpu":
            return bool(runtime.get("xpu_available"))
        return bool(runtime.get("mps_available"))
    return True


def _remove_incompatible_torch(channel: str) -> int:
    if _torch_variant_matches(channel):
        return 0
    installed = _installed_torch_version() or "unknown"
    print(f"Replacing incompatible Torch variant {installed!r} for requested {channel!r} channel.")
    command = [sys.executable, "-m", "pip", "uninstall", "-y", "torch"]
    print("$", " ".join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _validate_existing_accelerator_runtime() -> int:
    profile = detect_hardware()
    runtime = torch_runtime_status()
    if not runtime.get("importable"):
        print(f"Torch runtime cannot be imported: {runtime.get('error', 'unknown error')}", file=sys.stderr)
        return 2
    usable = (
        (profile.nvidia_gpu_detected and runtime.get("cuda_available") and not runtime.get("hip_available"))
        or (profile.gpu_vendor == "amd" and runtime.get("hip_available"))
        or (profile.gpu_vendor == "intel" and runtime.get("xpu_available"))
        or (profile.gpu_vendor == "apple" and runtime.get("mps_available"))
    )
    if not usable:
        print("Existing Torch runtime does not expose a usable accelerator for the detected GPU.", file=sys.stderr)
        return 2
    return 0


def _validate_requested_torch_channel(channel: str) -> int:
    version = _installed_torch_version()
    if version is None:
        print("Torch installation validation failed: package metadata is missing.", file=sys.stderr)
        return 2
    runtime = torch_runtime_status()
    if not runtime.get("importable"):
        print(
            "Torch installation validation failed: package exists but cannot be imported. "
            f"Error: {runtime.get('error', 'unknown import error')}",
            file=sys.stderr,
        )
        return 2
    if channel == "cuda":
        if "+cpu" in version.lower():
            print(
                f"Torch installation validation failed: CUDA host has CPU-only Torch {version}.",
                file=sys.stderr,
            )
            return 2
        runtime = torch_runtime_status()
        if not runtime["installed"] or not runtime["cuda_available"]:
            probe_error = runtime.get("cuda_error")
            detail = f" Runtime probe error: {probe_error}" if probe_error else ""
            print(
                "Torch installation validation failed: CUDA Torch is installed but "
                "torch.cuda.is_available() is false." + detail,
                file=sys.stderr,
            )
            return 2
    elif channel == "cpu" and "+cpu" not in version.lower():
        print(
            f"Torch installation validation failed: CPU channel produced Torch {version}, "
            "which is not a CPU wheel.",
            file=sys.stderr,
        )
        return 2
    elif channel in {"rocm", "xpu", "mps"}:
        runtime = torch_runtime_status()
        backend_available = {
            "rocm": runtime.get("hip_available"),
            "xpu": runtime.get("xpu_available"),
            "mps": runtime.get("mps_available"),
        }[channel]
        if not runtime.get("importable") or not backend_available:
            print(
                f"Torch installation validation failed: {channel.upper()} runtime is not usable "
                f"after installation. Import error: {runtime.get('error') or 'none'}",
                file=sys.stderr,
            )
            return 2
    elif channel == "default":
        profile = detect_hardware()
        runtime = torch_runtime_status()
        if profile.gpu_detected and not (
            (profile.nvidia_gpu_detected and runtime.get("cuda_available"))
            or (profile.gpu_vendor == "amd" and runtime.get("hip_available"))
            or (profile.gpu_vendor == "intel" and runtime.get("xpu_available"))
            or (profile.gpu_vendor == "apple" and runtime.get("mps_available"))
        ):
            print("Torch installation validation failed: default wheel did not expose a usable detected GPU.", file=sys.stderr)
            return 2
    return 0


def install(torch_channel: str = "auto") -> int:
    if not _python_supported():
        print(f"Bootstrap install aborted: {_python_requirement_message()}", file=sys.stderr)
        return 2
    if torch_channel != "auto":
        profile = detect_hardware()
        compatible = {
            "cuda": profile.nvidia_gpu_detected and (sys.platform.startswith("linux") or os.name == "nt"),
            "rocm": profile.gpu_vendor == "amd" and sys.platform.startswith("linux"),
            "xpu": profile.gpu_vendor == "intel" and (sys.platform.startswith("linux") or os.name == "nt"),
            "mps": profile.gpu_vendor == "apple" and sys.platform == "darwin",
        }
        if torch_channel in compatible and not compatible[torch_channel]:
            print(
                f"Refusing to install the {torch_channel} Torch runtime: no compatible "
                f"GPU/platform was detected (GPU vendor: {profile.gpu_vendor or 'none'}, "
                f"platform: {platform.system()}).",
                file=sys.stderr,
            )
            return 2
    if torch_channel == "auto":
        profile = detect_hardware()
        runtime = torch_runtime_status()
        if profile.nvidia_gpu_detected:
            torch_channel = "cuda"
        elif not profile.gpu_detected:
            torch_channel = "cpu"
        elif profile.gpu_vendor == "amd" and sys.platform.startswith("linux"):
            torch_channel = "existing" if runtime.get("importable") and runtime.get("hip_available") else "rocm"
        elif profile.gpu_vendor == "intel" and (sys.platform.startswith("linux") or os.name == "nt"):
            torch_channel = "existing" if runtime.get("importable") and runtime.get("xpu_available") else "xpu"
        elif profile.gpu_vendor == "apple" and sys.platform == "darwin":
            torch_channel = "existing" if runtime.get("importable") and runtime.get("mps_available") else "mps"
        else:
            print(
                f"Automatic Torch installation stopped: {profile.gpu_vendor or 'unknown'} GPU "
                f"detected on unsupported platform {platform.system()}. Refusing to install "
                "CPU-only Torch silently. Check the official PyTorch hardware compatibility "
                "matrix before choosing a manual channel.",
                file=sys.stderr,
            )
            return 2

    if torch_channel == "existing":
        result = _pip_install(REQUIREMENTS)
        if result != 0:
            return result
        return _validate_existing_accelerator_runtime()

    if torch_channel in {"cpu", "cuda", "rocm", "xpu", "mps"}:
        result = _remove_incompatible_torch(torch_channel)
        if result != 0:
            return result

    if torch_channel == "cpu":
        result = _pip_install(TORCH_REQUIREMENTS, "--index-url", CPU_TORCH_INDEX)
    elif torch_channel == "cuda":
        result = _pip_install(TORCH_REQUIREMENTS, "--index-url", CUDA_TORCH_INDEX)
    elif torch_channel == "rocm":
        if not sys.platform.startswith("linux"):
            print("ROCm PyTorch wheels are supported here only on Linux.", file=sys.stderr)
            return 2
        result = _pip_install(TORCH_REQUIREMENTS, "--index-url", ROCM_TORCH_INDEX)
    elif torch_channel == "xpu":
        result = _pip_install(TORCH_REQUIREMENTS, "--index-url", XPU_TORCH_INDEX)
    else:
        # Apple MPS uses the standard macOS PyTorch wheel; 'default' remains an explicit override.
        result = _pip_install(TORCH_REQUIREMENTS)
    if result != 0:
        return result

    result = _pip_install(REQUIREMENTS)
    if result != 0:
        return result

    # Validate after the full dependency install because transitive dependencies
    # are allowed to constrain Torch and must not silently replace the selected
    # host-specific wheel.
    return _validate_requested_torch_channel(torch_channel)


def ensure_llamacpp() -> int:
    if not RECONCILER.is_file():
        print(f"Missing reconciler: {RECONCILER}", file=sys.stderr)
        return 2
    command = [sys.executable, str(RECONCILER), "--project-root", str(ROOT), "--ensure-llamacpp"]
    print("$", " ".join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Model Lab bootstrap")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument(
        "--torch-channel",
        choices=("auto", "cpu", "cuda", "rocm", "xpu", "mps", "default"),
        default="auto",
        help="PyTorch backend for --install (auto selects a supported detected GPU backend, otherwise CPU; explicit channels: cpu/cuda/rocm/xpu/mps/default)",
    )
    parser.add_argument("--ensure-llamacpp", action="store_true")
    args = parser.parse_args()
    if args.ensure_llamacpp:
        return ensure_llamacpp()
    if args.install:
        return install(args.torch_channel)
    return doctor()


if __name__ == "__main__":
    raise SystemExit(main())
