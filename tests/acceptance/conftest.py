from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def filesystem_corpus() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "filesystem_corpus"


@pytest.fixture
def load_json():
    def _load(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))
    return _load
