"""Phase 10 CLI contract tests."""
from __future__ import annotations

import json
from pathlib import Path

import mlframework


def test_cli_parser_exposes_clone_and_run_commands():
    parser = mlframework.build_parser()
    for command in ("doctor", "smoke", "dataset", "train", "evaluate", "export", "infer", "status", "runs"):
        args = parser.parse_args([command])
        assert args.command == command


def test_cli_resolves_paths_from_entrypoint(monkeypatch):
    assert mlframework.ROOT == Path(mlframework.__file__).resolve().parent
    assert mlframework.DEFAULT_CONFIG == mlframework.ROOT / "config" / "pipeline_config.yaml"


def test_cpu_fallback_is_not_reported_as_gpu(monkeypatch):
    monkeypatch.setattr(mlframework, "_gpu", lambda: (False, False, "CPU fallback", None))
    monkeypatch.setattr(mlframework, "_torch_state", lambda: (True, False, "PyTorch CPU"))
    monkeypatch.setattr(mlframework, "_ram_gib", lambda: 16.0)
    monkeypatch.setattr(mlframework, "_disk_gib", lambda: 20.0)
    monkeypatch.setattr(mlframework, "_tokenizer_ready", lambda: (True, "ok"))
    monkeypatch.setattr(mlframework, "_dataset_ready", lambda: (True, "ok"))
    monkeypatch.setattr(mlframework, "_llamacpp_ok", lambda: (False, "not built"))
    monkeypatch.setattr(mlframework, "_git_ok", lambda: (True, "git"))
    monkeypatch.setattr(mlframework, "_configuration_ok", lambda: (True, "config"))

    assert mlframework.doctor() == 0


def test_default_gguf_prefers_q4_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(mlframework, "ROOT", tmp_path)
    gguf = tmp_path / "output" / "gguf" / "model-q4_k_m.gguf"
    gguf.parent.mkdir(parents=True)
    manifest = gguf.parent / "export_manifest.json"
    manifest.write_text(
        json.dumps({"artifacts": {"Q4_K_M": {"artifact": {"path": str(gguf)}}}}),
        encoding="utf-8",
    )
    assert mlframework._default_gguf() == gguf
