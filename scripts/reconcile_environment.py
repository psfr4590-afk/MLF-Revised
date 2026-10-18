#!/usr/bin/env python3
"""Reconcile native build prerequisites without duplicating project logic."""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--ensure-llamacpp", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if not args.ensure_llamacpp:
        return 0
    script = root / "scripts" / ("bootstrap_llama_cpp.ps1" if platform.system() == "Windows" else "bootstrap_llama_cpp.sh")
    if not script.is_file():
        print(f"Missing native bootstrap script: {script}", file=sys.stderr)
        return 2
    command = (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)] if platform.system() == "Windows" else ["bash", str(script)])
    print("Reconciling native llama.cpp toolchain...")
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode != 0:
        return result.returncode

    patcher = root / "scripts" / "patch_llama_cpp_tokenizer.py"
    if not patcher.is_file():
        print(f"Missing llama.cpp tokenizer compatibility patcher: {patcher}", file=sys.stderr)
        return 2
    print("Applying Model Lab llama.cpp tokenizer compatibility overlay...")
    return subprocess.run(
        [sys.executable, str(patcher), "--llamacpp-dir", str(root / "third_party" / "llama.cpp")],
        cwd=root,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
