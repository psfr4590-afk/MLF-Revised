"""Regression tests for release-gate identity checks."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_release_ui_probe_uses_project_version():
    source = (ROOT / "scripts" / "verify_release.py").read_text(encoding="utf-8")
    assert "_project_version()" in source
    assert 'response.json().get("version") == _project_version()' in source
    assert 'response.json().get("version") == "1.3.0"' not in source


def test_release_docs_preserve_historical_bad_tag_and_define_next_candidate():
    data = (ROOT / "pyproject.toml").read_bytes()
    version = tomllib.loads(data.decode("utf-8"))["project"]["version"]
    docs = (ROOT / "docs" / "release" / "RELEASE_READINESS_PLAN.md").read_text(encoding="utf-8")
    assert version == "1.3.0rc1"
    assert "v1.0.0-rc.1" in docs
    assert "v1.3.0-rc.1" in docs
    assert "must not be reused or force-moved" in docs


def test_release_readiness_docs_distinguish_native_smoke_from_full_rc():
    docs = (ROOT / "docs" / "release" / "RELEASE_READINESS_PLAN.md").read_text(encoding="utf-8")
    assert "python scripts/verify_release.py --bootstrap-native" in docs
    assert (
        "python scripts/verify_release.py --profile rc --bootstrap-native "
        "--ui-probe --clean-clone"
    ) in docs
    assert "does not implicitly run the full RC profile" in docs
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert (
        "verify_release.py --profile rc --bootstrap-native --ui-probe --clean-clone"
        in contributing
    )
