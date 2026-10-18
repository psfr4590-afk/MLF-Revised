"""Shared security invariants for local command-center trust boundaries."""
from __future__ import annotations

import os
from pathlib import Path

from command_center.config import ROOT

IMPORT_ROOT = ROOT / "imports"
MAX_INGEST_FILES = 10_000
MAX_INGEST_BYTES = 50 * 1024 * 1024 * 1024
MAX_LOG_LINES = 1_000
MAX_REQUEST_BYTES = 1_048_576


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def is_within(path: str | Path, root: str | Path) -> bool:
    try:
        resolved(path).relative_to(resolved(root))
        return True
    except ValueError:
        return False


def validate_import_source(path: str | Path) -> Path:
    """Validate a relative ingest path beneath IMPORT_ROOT.
    
    Absolute paths and parent traversal are rejected before filesystem
    resolution. Symlink sources are rejected so an allowed path cannot be
    redirected elsewhere after validation.
    """
    user_path = Path(path)
    if user_path.is_absolute() or ".." in user_path.parts:
        raise ValueError("ingest source must be inside the configured imports directory")

    root = IMPORT_ROOT.resolve(strict=False)
    candidate = root / user_path
    if candidate.is_symlink():
        raise ValueError("symlink ingest sources are not permitted")

    source = candidate.resolve(strict=False)
    if not is_within(source, root):
        raise ValueError("ingest source must be inside the configured imports directory")
    if not source.exists():
        raise FileNotFoundError(source)
    return source


def validate_log_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError("symlink log paths are not permitted")
    candidate = raw.resolve(strict=False)
    allowed_roots = (ROOT / "datasets", ROOT / ".runtime")
    if not any(is_within(candidate, root) for root in allowed_roots):
        raise ValueError("log path is outside the command-center log roots")
    if not candidate.is_file():
        raise ValueError("log path must be a regular file")
    return candidate


def validate_tail_lines(lines: int) -> int:
    value = int(lines)
    return max(1, min(value, MAX_LOG_LINES))


def reject_symlink_tree(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"symlinks are not permitted during ingest: {path}")
    if path.is_dir():
        for root, dirs, files in os.walk(path, followlinks=False):
            for name in [*dirs, *files]:
                candidate = Path(root) / name
                if candidate.is_symlink():
                    raise ValueError(f"symlinks are not permitted during ingest: {candidate}")
