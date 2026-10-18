#!/usr/bin/env python3
"""Runtime project-root discovery and environment-state helpers."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_project_root(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and all((path / x).is_file() if x in {"run_pipeline.py", "bootstrap.py", "requirements.txt"} else (path / x).is_dir() for x in ("run_pipeline.py", "bootstrap.py", "requirements.txt", "pipeline", "config"))


def discover_project_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (start, *start.parents):
        if is_project_root(candidate):
            return candidate
    queue: list[tuple[Path, int]] = [(start, 0)]
    seen: set[Path] = set()
    candidates: set[Path] = set()
    while queue:
        current, depth = queue.pop(0)
        if current in seen or depth > 3:
            continue
        seen.add(current)
        try:
            if is_project_root(current):
                candidates.add(current)
            if depth < 3:
                for child in current.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        queue.append((child, depth + 1))
        except (OSError, PermissionError):
            continue
    ordered = sorted(candidates, key=lambda p: (len(p.parts), str(p).lower()))
    if not ordered:
        raise RuntimeError(f"Unable to locate Model Lab project root from {start}")
    best = [p for p in ordered if len(p.parts) == len(ordered[0].parts)]
    if len(best) > 1:
        raise RuntimeError("Multiple possible pipeline roots found; refusing ambiguous execution:\n" + "\n".join(map(str, best)))
    return ordered[0]


def normalize_path(p: str | Path) -> str:
    return str(Path(p).resolve())


def executable_version(exe: Path) -> str | None:
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=10, check=False)
        lines = (r.stdout or r.stderr).strip().splitlines()
        return lines[0] if lines else None
    except Exception:
        return None


def all_executables(names: Iterable[str]) -> dict[str, dict]:
    result = {}
    for name in names:
        path = shutil.which(name)
        result[name] = {"status": "FOUND" if path else "MISSING", "path": normalize_path(path) if path else None, "version": executable_version(Path(path)) if path else None}
    return result


def python_state() -> dict:
    return {"status": "VERIFIED", "executable": normalize_path(sys.executable), "version": platform.python_version(), "implementation": platform.python_implementation()}


def write_environment_state(root: Path, state: dict) -> Path:
    d = root / ".runtime"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 2, "updated_at": utc_now(), "project_root": str(root.resolve()), "python": python_state(), "environment": {"os": platform.platform(), "cwd_at_probe": str(Path.cwd().resolve()), "path": os.environ.get("PATH", "")}, "tools": state.get("tools", {}), "python_packages": state.get("python_packages", {}), "capabilities": state.get("capabilities", {}), "events": state.get("events", [])}
    path = d / "environment.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path
