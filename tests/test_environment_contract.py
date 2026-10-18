from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import scripts.reconcile_environment as reconciler

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_reconciler_exists_and_is_referenced():
    script = ROOT / "scripts" / "reconcile_environment.py"
    assert script.is_file()
    source = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
    assert "reconcile_environment.py" in source
    assert "--ensure-llamacpp" in source


def test_reconciler_dispatches_to_powershell_on_windows(monkeypatch, tmp_path):
    script = tmp_path / "scripts" / "bootstrap_llama_cpp.ps1"
    patcher = tmp_path / "scripts" / "patch_llama_cpp_tokenizer.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture\n", encoding="utf-8")
    patcher.write_text("# fixture\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(reconciler.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        reconciler.subprocess,
        "run",
        lambda command, cwd, check: calls.append((command, cwd, check)) or SimpleNamespace(returncode=0),
    )

    assert reconciler.main(["--project-root", str(tmp_path), "--ensure-llamacpp"]) == 0
    assert calls == [
        (
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            tmp_path,
            False,
        ),
        (
            [reconciler.sys.executable, str(patcher), "--llamacpp-dir", str(tmp_path / "third_party" / "llama.cpp")],
            tmp_path,
            False,
        ),
    ]


def test_reconciler_does_not_apply_tokenizer_overlay_after_native_failure(monkeypatch, tmp_path):
    script = tmp_path / "scripts" / "bootstrap_llama_cpp.ps1"
    patcher = tmp_path / "scripts" / "patch_llama_cpp_tokenizer.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture\n", encoding="utf-8")
    patcher.write_text("# fixture\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(reconciler.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        reconciler.subprocess,
        "run",
        lambda command, cwd, check: calls.append((command, cwd, check)) or SimpleNamespace(returncode=9),
    )

    assert reconciler.main(["--project-root", str(tmp_path), "--ensure-llamacpp"]) == 9
    assert calls == [
        (
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            tmp_path,
            False,
        )
    ]


def test_reconciler_dispatches_to_bash_on_non_windows(monkeypatch, tmp_path):
    script = tmp_path / "scripts" / "bootstrap_llama_cpp.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(reconciler.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        reconciler.subprocess,
        "run",
        lambda command, cwd, check: calls.append((command, cwd, check)) or SimpleNamespace(returncode=7),
    )

    assert reconciler.main(["--project-root", str(tmp_path), "--ensure-llamacpp"]) == 7
    assert calls == [(["bash", str(script)], tmp_path, False)]


def test_runtime_requirements_include_command_center_security_dependencies():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package in ["fastapi", "uvicorn", "cryptography", "httpx", "pydantic"]:
        assert package in metadata


def test_semantic_acceleration_is_required_for_runtime():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "sentence-transformers" in metadata
    assert "faiss-cpu" in metadata


def test_native_bootstrap_has_platform_specific_build_profiles():
    source = (ROOT / "scripts/bootstrap_llama_cpp.sh").read_text(encoding="utf-8")
    assert "TERMUX_VERSION" in source
    assert "GGML_NATIVE=OFF" in source
    assert "GGML_OPENMP=OFF" in source
    assert "llama-quantize" in source
    assert 'BUILD_DIR="build-model-lab"' in source
    assert "b10516" in source
    assert "b95502b" in source

    android_start = source.index("# Android/Termux")
    android_end = source.index("else", android_start)
    android_profile = source[android_start:android_end]
    desktop_profile = source[android_end:]
    assert "--target llama-quantize" in android_profile
    assert "llama-cli" not in android_profile
    assert "--target llama-quantize llama-cli" in desktop_profile
    assert "llama-cli" in desktop_profile


def test_release_gate_cannot_skip_tests():
    source = (ROOT / "scripts/verify_release.py").read_text(encoding="utf-8")
    assert "--skip-tests" in source
    assert "not allowed for a production verification" in source
