from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PUBLIC_FILES = {
    "README.md",
    "docs/development/START_HERE.md",
    "LICENSE",
    ".gitignore",
    "pyproject.toml",
    "requirements.txt",
    "requirements-torch.txt",
    "bootstrap.py",
    "bootstrap_windows.ps1",
    "run_pipeline.py",
    "launch.py",
    "run_command_center.py",
    "scripts/verify_release.py",
}

FORBIDDEN_PUBLIC_PATH_PARTS = {
    ".venv",
    "venv",
    ".pytest_cache",
    ".ruff_cache",
    "datasets",
    "output",
    "scratch",
    "checkpoints",
    "models",
    "third_party",
}

# These paths are explicitly local-only in .gitignore. A source ZIP has no Git
# index, so they cannot be distinguished from tracked files by filesystem walk.
LOCAL_ONLY_PATH_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    ".runtime",
    "datasets",
    "output",
    "scratch",
    "checkpoints",
    "models",
    "third_party",
}


def _tracked_paths() -> set[str]:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # GitHub source ZIPs intentionally omit .git. Fall back to the public
        # filesystem tree while excluding paths known to be local-only.
        return {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and not any(part in LOCAL_ONLY_PATH_PARTS for part in path.relative_to(ROOT).parts)
        }
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_public_tree_contains_canonical_entrypoints() -> None:
    tracked = _tracked_paths()
    missing = sorted(REQUIRED_PUBLIC_FILES - tracked)
    assert not missing, f"Missing canonical public files: {missing}"


def test_public_tree_contains_no_generated_or_local_runtime_paths() -> None:
    tracked = _tracked_paths()
    violations = sorted(
        path
        for path in tracked
        if any(part in FORBIDDEN_PUBLIC_PATH_PARTS for part in Path(path).parts)
        or path.endswith((".log", ".tmp", ".ckpt", ".pt", ".pth", ".gguf"))
    )
    secret_paths = []
    for path in tracked:
        item = Path(path)
        name = item.name.lower()
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            secret_paths.append(path)
        elif name in {"credentials.yaml", "credentials.yml", "secrets.yaml", "secrets.yml"}:
            secret_paths.append(path)
        elif item.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            secret_paths.append(path)
    assert not violations, f"Generated/local artifacts are tracked publicly: {violations[:20]}"
    assert not secret_paths, f"Local secrets or credentials are tracked publicly: {sorted(secret_paths)[:20]}"


def test_canonical_first_run_commands_are_documented() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start_here = (ROOT / "docs/development/START_HERE.md").read_text(encoding="utf-8")
    combined = f"{readme}\n{start_here}"
    required_fragments = [
        "bootstrap.py --install",
        "bootstrap.py --doctor",
        "run_pipeline.py --no-resume",
        "config/pipeline_config.yaml",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in combined]
    assert not missing, f"Canonical first-run documentation is incomplete: {missing}"


def test_no_large_local_vendor_checkout_is_tracked() -> None:
    tracked = _tracked_paths()
    vendor_paths = [path for path in tracked if path.startswith("third_party/llama.cpp/")]
    assert not vendor_paths, "llama.cpp source should remain a separately bootstrapped dependency"


@pytest.mark.parametrize(
    "command",
    [
        "python bootstrap.py --install",
        "python bootstrap.py --doctor",
        "python run_pipeline.py --list-stages",
        "python run_pipeline.py --list-groups",
        "python run_pipeline.py --doctor",
        "python run_pipeline.py --no-resume",
    ],
)
def test_documented_commands_reference_existing_entrypoints(command: str) -> None:
    entrypoint = command.split()[1]
    assert (ROOT / entrypoint).is_file(), f"Documented entrypoint does not exist: {entrypoint}"


def test_windows_first_boot_wrapper_uses_canonical_bootstrap_safely() -> None:
    source = (ROOT / "bootstrap_windows.ps1").read_text(encoding="utf-8-sig")
    required_fragments = [
        "[switch]$Native",
        "function Find-Python",
        '@{ Command = "py"; Args = @("-3.13") }',
        "versionText -match '^3\\.(11|12|13|14)$'",
        "Python.Python.3.13",
        '"--install"',
        '"--doctor"',
        '"--ensure-llamacpp"',
        "Project dependency installation failed",
        "Native llama.cpp setup failed",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    assert not missing, f"Windows first-boot contract is incomplete: {missing}"
    assert "Remove-Item" not in source, "First-boot setup must not delete user or project data"
