#!/usr/bin/env python3
"""Verify that a GGUF artifact is structurally sound, loads, and generates with llama.cpp.

Checks, in order: GGUF header, metadata, (optionally) the tensor table and
tensor-data completeness, architecture/context/tokenizer/special-token
metadata, then real noninteractive generation through llama.cpp. A zero exit
code alone is never treated as success.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

# (elements per block, bytes per block) for ggml tensor types.
GGML_TYPES: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 24: (1, 1), 25: (1, 2),
    26: (1, 4), 27: (1, 8), 28: (1, 8), 30: (1, 2),
}
DEFAULT_ALIGNMENT = 32
LAYER_TENSORS = (
    "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
    "ffn_norm", "ffn_gate", "ffn_down", "ffn_up",
)
SCALAR_FORMATS = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
    10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}


class GGUFError(RuntimeError):
    """Raised for malformed, truncated, or inconsistent GGUF files."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _find_cli(llamacpp_dir: Path) -> Path:
    """Find a llama.cpp inference executable across supported build layouts."""
    names = ("llama.exe", "llama-cli.exe", "llama-cli")
    bases = (llamacpp_dir / "build", llamacpp_dir / "build-model-lab", llamacpp_dir / "bin", llamacpp_dir)
    for base in bases:
        if not base.is_dir():
            continue
        for name in names:
            matches = list(base.rglob(name))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"llama inference executable not found under {llamacpp_dir}")


class _Reader:
    """Bounds-checked little-endian reader: truncation and absurd lengths raise GGUFError."""

    def __init__(self, handle, size: int):
        self.h = handle
        self.size = size

    def tell(self) -> int:
        return self.h.tell()

    def read(self, n: int) -> bytes:
        if n < 0 or n > self.size - self.h.tell():
            raise GGUFError(f"truncated or corrupt GGUF: cannot read {n} bytes at offset {self.h.tell()}")
        raw = self.h.read(n)
        if len(raw) != n:
            raise GGUFError("truncated GGUF")
        return raw

    def unpack(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read(size))

    def string(self) -> str:
        length = self.unpack("Q")[0]
        raw = self.read(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("corrupt GGUF string (invalid UTF-8)") from exc

    def value(self, kind: int) -> Any:
        if kind == 8:
            return self.string()
        if kind == 9:
            element_type = self.unpack("I")[0]
            count = self.unpack("Q")[0]
            if count > self.size:  # cannot possibly be satisfied by this file
                raise GGUFError(f"corrupt GGUF array length {count}")
            return [self.value(element_type) for _ in range(count)]
        fmt = SCALAR_FORMATS.get(kind)
        if fmt is None:
            raise GGUFError(f"unsupported GGUF metadata type: {kind}")
        return self.unpack(fmt[0])[0]


def _tensor_nbytes(name: str, dims: list[int], ggml_type: int) -> int:
    if ggml_type not in GGML_TYPES:
        raise GGUFError(f"tensor {name!r} has unsupported ggml type {ggml_type}")
    block, size = GGML_TYPES[ggml_type]
    elements = 1
    for d in dims:
        elements *= d
    if elements % block:
        raise GGUFError(f"tensor {name!r} element count {elements} is not a multiple of block size {block}")
    return elements // block * size


def _parse(path: Path, *, with_tensors: bool) -> dict[str, Any]:
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        r = _Reader(handle, file_size)
        if r.read(4) != b"GGUF":
            raise GGUFError(f"invalid GGUF magic: {path}")
        version = r.unpack("I")[0]
        if version not in (1, 2, 3):
            raise GGUFError(f"unsupported GGUF version: {version}")
        tensor_count, metadata_count = r.unpack("QQ")
        if tensor_count > file_size or metadata_count > file_size:
            raise GGUFError("corrupt GGUF header counts")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = r.string()
            kind = r.unpack("I")[0]
            metadata[key] = r.value(kind)
        result: dict[str, Any] = {
            "version": version, "tensor_count": tensor_count,
            "metadata_count": metadata_count, "metadata": metadata,
            "file_size": file_size,
        }
        if not with_tensors:
            return result

        tensors: dict[str, dict[str, Any]] = {}
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.unpack("I")[0]
            if not 1 <= n_dims <= 8:
                raise GGUFError(f"tensor {name!r} has invalid dimension count {n_dims}")
            dims = list(r.unpack("Q" * n_dims))
            ggml_type = r.unpack("I")[0]
            offset = r.unpack("Q")[0]
            if name in tensors:
                raise GGUFError(f"duplicate tensor name {name!r}")
            tensors[name] = {"dims": dims, "type": ggml_type, "offset": offset,
                             "nbytes": _tensor_nbytes(name, dims, ggml_type)}
        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT)) or DEFAULT_ALIGNMENT
        data_start = (r.tell() + alignment - 1) // alignment * alignment
        for name, t in tensors.items():
            if t["offset"] % alignment:
                raise GGUFError(f"tensor {name!r} offset {t['offset']} is not {alignment}-byte aligned")
            end = data_start + t["offset"] + t["nbytes"]
            if end > file_size:
                raise GGUFError(
                    f"truncated tensor data: {name!r} needs bytes up to {end} but file has {file_size}"
                )
        result.update({"tensors": tensors, "data_start": data_start, "alignment": alignment})
        return result


def _read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Read GGUF header + metadata only (no dependency on the ``gguf`` package)."""
    return _parse(path, with_tensors=False)


def read_gguf_structure(path: Path) -> dict[str, Any]:
    """Read header, metadata, tensor table, and verify tensor data fits in the file."""
    return _parse(path, with_tensors=True)


def validate_llama_tensors(structure: dict[str, Any]) -> dict[str, Any]:
    """Check the tensor table matches the llama architecture described by the metadata."""
    md, tensors = structure["metadata"], structure["tensors"]
    if int(structure["tensor_count"]) != len(tensors):
        raise GGUFError(f"tensor count {structure['tensor_count']} != parsed {len(tensors)}")
    layers = md.get("llama.block_count")
    if layers is None:
        raise GGUFError("GGUF does not expose llama.block_count")
    required = ["token_embd.weight", "output_norm.weight"]
    for i in range(int(layers)):
        required += [f"blk.{i}.{part}.weight" for part in LAYER_TENSORS]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise GGUFError(f"GGUF is missing required tensors: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    embd = tensors["token_embd.weight"]["dims"]
    hidden = md.get("llama.embedding_length")
    if hidden is not None and int(embd[0]) != int(hidden):
        raise GGUFError(f"token_embd width {embd[0]} != llama.embedding_length {hidden}")
    tokens = md.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list) and len(embd) > 1 and int(embd[1]) != len(tokens):
        raise GGUFError(f"token_embd rows {embd[1]} != tokenizer vocabulary size {len(tokens)}")
    return {"tensors_checked": len(tensors), "layers": int(layers),
            "vocab_size": int(embd[1]) if len(embd) > 1 else None}


def _build_command(
    cli_path: Path,
    model_path: Path,
    prompt: str,
    context_length: int | None = None,
    *,
    deterministic: bool = True,
    seed: int = 42,
) -> list[str]:
    command = [
        str(cli_path), "-m", str(model_path), "-p", prompt, "-n", "1",
        "--single-turn", "--no-display-prompt", "--simple-io", "--check-tensors",
    ]
    if deterministic:
        command.extend(["--temp", "0", "--seed", str(seed)])
    if context_length is not None:
        command.extend(["-c", str(context_length)])
    if cli_path.name.lower() == "llama.exe":
        return [str(cli_path), "cli", *command[1:]]
    return command


def _run_generation(cmd: list[str]) -> tuple[str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=120)
    stdout, stderr = proc.stdout.strip(), proc.stderr.strip()
    if proc.returncode != 0:
        detail = stderr[-2000:] if stderr else stdout[-2000:]
        raise RuntimeError(f"llama inference failed with exit code {proc.returncode}: {detail}")
    if not stdout:
        raise RuntimeError("llama.cpp loaded the GGUF but generated no visible token output")
    return stdout, stderr


def verify_gguf(
    model: str | Path,
    llama_cli: str | Path,
    prompt: str = "Hello",
    context_length: int | None = None,
    expected_special_tokens: dict[str, int] | None = None,
    *,
    check_tensor_structure: bool = False,
    repeat: int = 1,
    evidence_dir: str | Path | None = None,
) -> dict[str, Any]:
    model_path = Path(model).resolve()
    cli_path = Path(llama_cli).resolve()

    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise FileNotFoundError(f"GGUF model is missing or empty: {model_path}")
    if not cli_path.is_file():
        raise FileNotFoundError(f"llama inference executable is missing: {cli_path}")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    structure = _parse(model_path, with_tensors=check_tensor_structure)
    values = structure["metadata"]
    tensor_report = validate_llama_tensors(structure) if check_tensor_structure else None
    architecture = values.get("general.architecture")
    if architecture not in (None, "llama"):
        raise RuntimeError(f"unexpected GGUF architecture: {architecture!r}")

    context_key = f"{architecture or 'llama'}.context_length"
    embedded_context = values.get(context_key)
    if context_length is not None:
        if embedded_context is None:
            raise RuntimeError(f"GGUF does not expose {context_key}")
        if int(embedded_context) != int(context_length):
            raise RuntimeError(
                f"GGUF context length mismatch: expected {context_length}, embedded {embedded_context}"
            )

    special_tokens = {
        "bos": values.get("tokenizer.ggml.bos_token_id"),
        "eos": values.get("tokenizer.ggml.eos_token_id"),
        "pad": values.get("tokenizer.ggml.padding_token_id"),
    }
    if expected_special_tokens:
        for name, expected in expected_special_tokens.items():
            actual = special_tokens.get(name)
            if actual is None:
                raise RuntimeError(f"GGUF is missing expected {name} special-token metadata")
            if int(actual) != int(expected):
                raise RuntimeError(f"GGUF special token mismatch for {name}: expected {expected}, embedded {actual}")
    if not any(k.startswith("tokenizer.ggml.") for k in values):
        raise RuntimeError("GGUF tokenizer metadata is missing")

    cmd = _build_command(cli_path, model_path, prompt, context_length)
    stdout, stderr = _run_generation(cmd)
    for _ in range(repeat - 1):
        again, _ = _run_generation(cmd)
        if again != stdout:
            raise RuntimeError(
                "non-deterministic generation: identical greedy runs produced different output "
                f"({stdout!r} vs {again!r})"
            )

    result = {
        "status": "passed",
        "model": str(model_path),
        "llama_cli": str(cli_path),
        "command": cmd,
        "prompt": prompt,
        "generated_output": stdout,
        "checks": {
            "gguf_header": "passed",
            "model_loading": "passed",
            "tensor_validation": "passed",
            "tensor_structure": tensor_report if tensor_report else "not_checked",
            "architecture": architecture or "llama",
            "context_length": embedded_context if embedded_context is not None else context_length,
            "tokenizer": "passed",
            "special_tokens": special_tokens,
            "special_token_validation": "passed" if expected_special_tokens else "recorded",
            "generation": "passed",
            "repeat_runs": repeat,
            "repeat_run_compared": repeat > 1,
        },
        "metadata": {
            "version": structure["version"],
            "tensor_count": structure["tensor_count"],
            "metadata_count": structure["metadata_count"],
        },
    }
    if evidence_dir is not None:
        out = Path(evidence_dir)
        out.mkdir(parents=True, exist_ok=True)
        _atomic_json(out / "inference-smoke-test.json", result)
        (out / "inference-smoke-test.txt").write_text(
            f"status: passed\nmodel: {model_path}\nprompt: {prompt}\noutput: {stdout}\n", encoding="utf-8"
        )
        (out / "llama-cli.log").write_text(stderr + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load and execute a GGUF with llama.cpp")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-cli", required=False)
    parser.add_argument("--llamacpp-dir", required=False)
    parser.add_argument("--manifest")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--bos-token-id", type=int)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--pad-token-id", type=int)
    parser.add_argument("--repeat", type=int, default=2, help="identical greedy runs that must match (default 2)")
    parser.add_argument("--evidence-dir", help="write inference-smoke-test.{json,txt} and llama-cli.log here")
    parser.add_argument("--no-tensor-structure-check", action="store_true",
                        help="skip tensor-table/data-completeness validation (not for release use)")
    args = parser.parse_args(argv)

    if bool(args.llama_cli) == bool(args.llamacpp_dir):
        parser.error("provide exactly one of --llama-cli or --llamacpp-dir")

    try:
        cli = Path(args.llama_cli).resolve() if args.llama_cli else _find_cli(Path(args.llamacpp_dir).resolve())
        expected = {
            name: value for name, value in (
                ("bos", args.bos_token_id), ("eos", args.eos_token_id), ("pad", args.pad_token_id)
            ) if value is not None
        }
        result = verify_gguf(
            args.model, cli, args.prompt, args.context_length, expected,
            check_tensor_structure=not args.no_tensor_structure_check,
            repeat=args.repeat, evidence_dir=args.evidence_dir,
        )
        if args.manifest:
            manifest_path = Path(args.manifest).resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            manifest["inference_validation"] = result
            _atomic_json(manifest_path, manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"GGUF VERIFICATION FAILED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
