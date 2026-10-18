"""Canonical launcher for the single Model Lab Command Center."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMAND_CENTER = ROOT / "run_command_center.py"


def main() -> int:
    """Start the one operator interface: the local browser Command Center."""
    if not COMMAND_CENTER.is_file():
        raise SystemExit(f"run_command_center.py is missing: {COMMAND_CENTER}")
    return subprocess.call([sys.executable, str(COMMAND_CENTER)], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
