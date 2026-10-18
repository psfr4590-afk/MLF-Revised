#!/usr/bin/env python3
"""Generate reproducibility evidence for a release environment."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import tomllib
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release-evidence"


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_evidence() -> dict:
    root = ROOT / "third_party" / "llama.cpp"
    if not (root / ".git").is_dir():
        return {"present": False}
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    tools = {}
    for name in ("llama-cli", "llama-quantize"):
        matches = list(root.rglob(name)) + list(root.rglob(name + ".exe"))
        if matches:
            tools[name] = {"path": str(matches[0].relative_to(ROOT)).replace("\\\\", "/"), "sha256": _sha256(matches[0])}
    cmake = subprocess.run(["cmake", "--version"], capture_output=True, text=True, check=False)
    return {
        "present": True,
        "tag": git("describe", "--tags", "--exact-match"),
        "commit": git("rev-parse", "HEAD"),
        "converter_sha256": _sha256(root / "convert_hf_to_gguf.py") if (root / "convert_hf_to_gguf.py").is_file() else None,
        "tools": tools,
        "cmake": (cmake.stdout or cmake.stderr).splitlines()[0] if (cmake.stdout or cmake.stderr) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true", help="include pinned llama.cpp and native binary evidence")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=ROOT, check=True, capture_output=True, text=True)
    (OUT / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    run("cyclonedx-py", "environment", "-o", str(OUT / "sbom.json"), "--format", "json")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    (OUT / "release_metadata.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit,
                "project_version": project["project"]["version"],
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "status": "dependency-evidence-generated",
                "native_toolchain": native_evidence() if args.native else {"present": False, "requested": False},
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
