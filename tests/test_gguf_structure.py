"""GGUF structural verification: truncation, tensor table, determinism, evidence."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from scripts import verify_gguf as vg

ALIGN = 32
HIDDEN, VOCAB, LAYERS = 4, 8, 1


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, kind: int, value) -> bytes:
    body = b""
    if kind == 8:
        body = _s(value)
    elif kind == 4:
        body = struct.pack("<I", value)
    elif kind == 9:  # array of strings
        body = struct.pack("<IQ", 8, len(value)) + b"".join(_s(v) for v in value)
    return _s(key) + struct.pack("<I", kind) + body


def build_gguf(
    path: Path,
    *,
    drop_tensor: str | None = None,
    embd_rows: int = VOCAB,
    truncate_bytes: int = 0,
    extra_meta: list[bytes] | None = None,
) -> Path:
    tensors: list[tuple[str, list[int]]] = [
        ("token_embd.weight", [HIDDEN, embd_rows]),
        ("output_norm.weight", [HIDDEN]),
    ]
    for i in range(LAYERS):
        for part in vg.LAYER_TENSORS:
            dims = [HIDDEN] if part.endswith("norm") else [HIDDEN, HIDDEN]
            tensors.append((f"blk.{i}.{part}.weight", dims))
    tensors = [t for t in tensors if t[0] != drop_tensor]

    meta = [
        _kv("general.architecture", 8, "llama"),
        _kv("llama.block_count", 4, LAYERS),
        _kv("llama.embedding_length", 4, HIDDEN),
        _kv("llama.context_length", 4, 16),
        _kv("tokenizer.ggml.model", 8, "gpt2"),
        _kv("tokenizer.ggml.tokens", 9, [f"t{i}" for i in range(VOCAB)]),
        _kv("tokenizer.ggml.bos_token_id", 4, 2),
        _kv("tokenizer.ggml.eos_token_id", 4, 3),
        _kv("tokenizer.ggml.padding_token_id", 4, 0),
    ] + (extra_meta or [])

    infos, offset = b"", 0
    for name, dims in tensors:
        n = 4
        for d in dims:
            n *= d  # f32
        infos += _s(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        infos += struct.pack("<IQ", 0, offset)
        offset += (n + ALIGN - 1) // ALIGN * ALIGN
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + b"".join(meta) + infos
    head += b"\0" * ((-len(head)) % ALIGN)
    data = head + b"\0" * offset
    path.write_bytes(data[: len(data) - truncate_bytes] if truncate_bytes else data)
    return path


class _Proc:
    def __init__(self, out="hi", code=0):
        self.returncode, self.stdout, self.stderr = code, out, "log"


def _cli(tmp_path: Path) -> Path:
    cli = tmp_path / "llama-cli"
    cli.write_bytes(b"exe")
    return cli


def test_valid_gguf_structure_passes(tmp_path):
    structure = vg.read_gguf_structure(build_gguf(tmp_path / "m.gguf"))
    report = vg.validate_llama_tensors(structure)
    assert report == {"tensors_checked": 2 + 9 * LAYERS, "layers": LAYERS, "vocab_size": VOCAB}


def test_truncated_tensor_data_is_detected(tmp_path):
    path = build_gguf(tmp_path / "m.gguf", truncate_bytes=64)
    with pytest.raises(vg.GGUFError, match="truncated tensor data"):
        vg.read_gguf_structure(path)


def test_missing_required_tensor_is_detected(tmp_path):
    structure = vg.read_gguf_structure(build_gguf(tmp_path / "m.gguf", drop_tensor="blk.0.attn_q.weight"))
    with pytest.raises(vg.GGUFError, match="missing required tensors"):
        vg.validate_llama_tensors(structure)


def test_embedding_rows_must_match_tokenizer_vocabulary(tmp_path):
    structure = vg.read_gguf_structure(build_gguf(tmp_path / "m.gguf", embd_rows=VOCAB + 1))
    with pytest.raises(vg.GGUFError, match="vocabulary size"):
        vg.validate_llama_tensors(structure)


def test_bad_magic_wrong_version_and_empty_header_fail(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(vg.GGUFError, match="magic"):
        vg.read_gguf_structure(bad)
    bad.write_bytes(b"GGUF" + struct.pack("<IQQ", 99, 0, 0))
    with pytest.raises(vg.GGUFError, match="version"):
        vg.read_gguf_structure(bad)
    bad.write_bytes(b"GGUF")
    with pytest.raises(vg.GGUFError, match="truncated"):
        vg.read_gguf_structure(bad)


def test_absurd_string_length_does_not_allocate_or_crash(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**60) + b"abc")
    with pytest.raises(vg.GGUFError):
        vg.read_gguf_structure(bad)


def test_verify_gguf_strict_mode_rejects_truncated_model_before_running_cli(tmp_path, monkeypatch):
    model = build_gguf(tmp_path / "m.gguf", truncate_bytes=64)
    called = []
    monkeypatch.setattr(vg.subprocess, "run", lambda *a, **k: called.append(1) or _Proc())
    with pytest.raises(vg.GGUFError):
        vg.verify_gguf(model, _cli(tmp_path), check_tensor_structure=True)
    assert not called


def test_verify_gguf_passes_with_structure_context_and_special_tokens(tmp_path, monkeypatch):
    model = build_gguf(tmp_path / "m.gguf")
    seen = {}
    monkeypatch.setattr(vg.subprocess, "run", lambda cmd, **k: seen.setdefault("cmd", cmd) and _Proc())
    result = vg.verify_gguf(
        model, _cli(tmp_path), context_length=16,
        expected_special_tokens={"bos": 2, "eos": 3, "pad": 0}, check_tensor_structure=True,
    )
    assert result["checks"]["tensor_structure"]["layers"] == LAYERS
    assert "--temp" in seen["cmd"] and "--seed" in seen["cmd"]


def test_context_and_special_token_mismatches_fail(tmp_path, monkeypatch):
    model = build_gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(vg.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="context length mismatch"):
        vg.verify_gguf(model, _cli(tmp_path), context_length=1024)
    with pytest.raises(RuntimeError, match="special token mismatch for eos"):
        vg.verify_gguf(model, _cli(tmp_path), expected_special_tokens={"eos": 99})


def test_repeat_runs_must_match_and_are_recorded(tmp_path, monkeypatch):
    model = build_gguf(tmp_path / "m.gguf")
    outputs = iter(["a", "b"])
    monkeypatch.setattr(vg.subprocess, "run", lambda *a, **k: _Proc(next(outputs)))
    with pytest.raises(RuntimeError, match="non-deterministic"):
        vg.verify_gguf(model, _cli(tmp_path), repeat=2)

    monkeypatch.setattr(vg.subprocess, "run", lambda *a, **k: _Proc("same"))
    result = vg.verify_gguf(model, _cli(tmp_path), repeat=2)
    assert result["checks"]["repeat_run_compared"] is True


def test_whitespace_only_generation_and_nonzero_exit_are_failures(tmp_path, monkeypatch):
    model = build_gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(vg.subprocess, "run", lambda *a, **k: _Proc("\n \n"))
    with pytest.raises(RuntimeError, match="no visible token output"):
        vg.verify_gguf(model, _cli(tmp_path))
    monkeypatch.setattr(vg.subprocess, "run", lambda *a, **k: _Proc("text", code=3))
    with pytest.raises(RuntimeError, match="exit code 3"):
        vg.verify_gguf(model, _cli(tmp_path))


def test_evidence_files_are_written(tmp_path, monkeypatch):
    model = build_gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(vg.subprocess, "run", lambda *a, **k: _Proc("tok"))
    evidence = tmp_path / "evidence"
    vg.verify_gguf(model, _cli(tmp_path), evidence_dir=evidence)
    assert json.loads((evidence / "inference-smoke-test.json").read_text())["status"] == "passed"
    assert (evidence / "inference-smoke-test.txt").is_file() and (evidence / "llama-cli.log").is_file()


def test_repeat_runs_ignore_variable_timing_banner(tmp_path, monkeypatch):
    model = build_gguf(tmp_path / "m.gguf")
    outputs = iter([
        "Loading model...\n> Hello\n4\n"
        "[ Prompt: 1138.8 t/s | Generation: 0.0 t/s ]\nExiting...",
        "Loading model...\n> Hello\n4\n"
        "[ Prompt: 1230.0 t/s | Generation: 0.0 t/s ]\nExiting...",
    ])
    monkeypatch.setattr(
        vg.subprocess, "run", lambda *args, **kwargs: _Proc(next(outputs))
    )

    result = vg.verify_gguf(model, _cli(tmp_path), repeat=2)

    assert result["checks"]["repeat_run_compared"] is True
    assert "1138.8" in result["generated_output"]
