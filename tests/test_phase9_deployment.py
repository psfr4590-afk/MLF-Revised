from __future__ import annotations

import struct
import subprocess
from pathlib import Path

from scripts.export_gguf import ALL_QUANTS, EXPORTER_VERSION, _artifact_manifest, _write_modelfile
from scripts.verify_gguf import _read_gguf_metadata
from scripts.verify_ollama import verify_ollama


def _write_gguf(path: Path, metadata: list[tuple[str, int, object]]) -> None:
    def string(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def value(kind: int, item: object) -> bytes:
        if kind == 8:
            return string(str(item))
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        return struct.pack("<" + formats[kind], item)

    payload = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, kind, item in metadata:
        payload += string(key) + struct.pack("<I", kind) + value(kind, item)
    path.write_bytes(payload + b"fixture")


def test_all_export_quantizations_are_explicit():
    assert ALL_QUANTS == ("F16", "Q4_K_M", "Q5_K_M", "Q8_0")
    assert EXPORTER_VERSION.startswith("9.")


def test_per_export_manifest_contains_required_release_evidence(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    artifact = tmp_path / "model-q4_k_m.gguf"
    hf = tmp_path / "hf"
    checkpoint.write_bytes(b"checkpoint")
    artifact.write_bytes(b"gguf")
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")
    record = _artifact_manifest(
        quant="Q4_K_M",
        artifact=artifact,
        checkpoint=checkpoint,
        payload={"step": 10},
        cfg=type("Cfg", (), {"to_dict": lambda self: {"seq_len": 128}})(),
        export_config={"quantizations": ["Q4_K_M"]},
        lineage={"run_id": "run-1"},
        converter_version="b10516@b95502b",
        hf_dir=hf,
    )
    assert record["checkpoint_sha256"]
    assert record["model_config"]["seq_len"] == 128
    assert record["export_config"]["quantizations"] == ["Q4_K_M"]
    assert record["exporter_version"].startswith("9.")
    assert record["artifact"]["sha256"]
    assert record["timestamp"].endswith("Z")
    assert record["lineage"]["run_id"] == "run-1"


def test_modelfile_references_exact_gguf(tmp_path: Path):
    gguf = tmp_path / "model-q4_k_m.gguf"
    modelfile = tmp_path / "Modelfile"
    gguf.write_bytes(b"gguf")
    _write_modelfile(modelfile, gguf)
    assert modelfile.read_text(encoding="utf-8").splitlines()[0] == "FROM model-q4_k_m.gguf"


def test_gguf_metadata_reader_extracts_context_and_special_tokens(tmp_path: Path):
    path = tmp_path / "model.gguf"
    _write_gguf(
        path,
        [
            ("general.architecture", 8, "llama"),
            ("llama.context_length", 4, 128),
            ("tokenizer.ggml.bos_token_id", 4, 2),
            ("tokenizer.ggml.eos_token_id", 4, 3),
            ("tokenizer.ggml.padding_token_id", 4, 0),
        ],
    )
    metadata = _read_gguf_metadata(path)
    assert metadata["metadata"]["general.architecture"] == "llama"
    assert metadata["metadata"]["llama.context_length"] == 128
    assert metadata["metadata"]["tokenizer.ggml.bos_token_id"] == 2


def test_ollama_verification_contract(tmp_path: Path, monkeypatch):
    model = tmp_path / "model-q4_k_m.gguf"
    modelfile = tmp_path / "Modelfile"
    model.write_bytes(b"gguf")
    modelfile.write_text("FROM model-q4_k_m.gguf\nPARAMETER temperature 0.7\n", encoding="utf-8")
    monkeypatch.setattr("scripts.verify_ollama.shutil.which", lambda _: "/usr/bin/ollama")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="generated", stderr="")

    monkeypatch.setattr("scripts.verify_ollama.subprocess.run", fake_run)
    monkeypatch.setattr(Path, "symlink_to", lambda self, target: self.write_bytes(Path(target).read_bytes()))
    result = verify_ollama(model, modelfile, model_name="model-lab-test")
    assert result["status"] == "passed"
    assert result["checks"]["ollama_create"] == "passed"
    assert result["checks"]["generation"] == "passed"
    assert [c[1] for c in calls] == ["create", "run", "show"]
