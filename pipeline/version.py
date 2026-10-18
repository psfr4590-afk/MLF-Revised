"""Resolve the canonical Model Lab package version."""
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def get_version() -> str:
    project_file = ROOT / "pyproject.toml"
    if project_file.is_file():
        with project_file.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])

    try:
        return distribution_version("model-lab-framework")
    except PackageNotFoundError:
        return "0+unknown"
