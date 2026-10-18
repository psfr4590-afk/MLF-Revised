"""Artifact integrity, provenance, and atomic-write helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = 2


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def write_manifest(
    path: Path,
    *,
    kind: str,
    rows: int | None = None,
    provenance: dict[str, Any] | None = None,
    extra: dict | None = None,
) -> Path:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "kind": kind,
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        payload["rows"] = rows
    if provenance:
        payload["provenance"] = provenance
    if extra:
        payload.update(extra)
    mp = manifest_path(path)
    tmp = mp.with_suffix(mp.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, mp)
    return mp


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file using a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def artifact_valid(path: Path, expected_provenance: dict[str, Any] | None = None) -> bool:
    """Return true only when bytes and semantic manifest metadata are valid."""
    if not path.exists() or path.stat().st_size <= 0:
        return False
    mp = manifest_path(path)
    if not mp.exists():
        return False
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
        if int(m.get("schema", -1)) != MANIFEST_SCHEMA or not isinstance(m.get("kind"), str) or not m["kind"]:
            return False
        if int(m.get("size", -1)) != path.stat().st_size:
            return False
        if m.get("sha256") != sha256_file(path):
            return False
        provenance = m.get("provenance") or {}
        if not isinstance(provenance, dict):
            return False
        identities = provenance.get("config_identities")
        bundle = provenance.get("identity_bundle_sha256")
        if identities is not None or bundle is not None:
            if not isinstance(identities, dict) or not isinstance(bundle, str):
                return False
            canonical = json.dumps(identities, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != bundle:
                return False
        if expected_provenance is not None and provenance != expected_provenance:
            return False
        source = m.get("source")
        source_sha256 = m.get("source_sha256")
        if (source is None) != (source_sha256 is None):
            return False
        if source is not None:
            if not isinstance(source, str) or not isinstance(source_sha256, str) or not source_sha256:
                return False
            source_path = Path(source)
            if not source_path.is_file() or sha256_file(source_path) != source_sha256:
                return False
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def checkpoint_manifest_valid(path: Path) -> bool:
    """Torch-free integrity check for a checkpoint and its sidecar manifest.

    Checkpoint manifests use their own, newer schema than generic artifact
    manifests, so ``artifact_valid`` (strict ``schema == 2``) must not be used
    for them. Accepts any schema >= MANIFEST_SCHEMA with kind ``checkpoint``.
    """
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        meta = json.loads(manifest_path(path).read_text(encoding="utf-8"))
        return (
            int(meta.get("schema", -1)) >= MANIFEST_SCHEMA
            and meta.get("kind") == "checkpoint"
            and int(meta.get("size", -1)) == path.stat().st_size
            and meta.get("sha256") == sha256_file(path)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def select_export_checkpoint(ckpt_dir: Path) -> Path:
    """Pick the checkpoint to export using the trainer's real file naming.

    Order matches the orchestrator: ``ckpt_final.pt``, ``ckpt_best.pt``, then the
    highest numbered ``ckpt_<step>.pt``. Every candidate must pass
    ``checkpoint_manifest_valid``; invalid candidates are skipped, never exported.
    Legacy step-suffixed ``ckpt_final_<step>.pt`` / ``ckpt_best_<step>.pt`` names
    are accepted only as a last resort.
    """
    ckpt_dir = Path(ckpt_dir)
    for name in ("ckpt_final.pt", "ckpt_best.pt"):
        candidate = ckpt_dir / name
        if checkpoint_manifest_valid(candidate):
            return candidate
    numbered = sorted(
        ((int(p.stem[5:]), p) for p in ckpt_dir.glob("ckpt_[0-9]*.pt") if p.stem[5:].isdigit()),
        key=lambda item: item[0], reverse=True,
    )
    for _, candidate in numbered:
        if checkpoint_manifest_valid(candidate):
            return candidate
    for pattern in ("ckpt_final_*.pt", "ckpt_best_*.pt"):
        legacy = sorted(
            (p for p in ckpt_dir.glob(pattern) if p.stem.rsplit("_", 1)[-1].isdigit()),
            key=lambda p: int(p.stem.rsplit("_", 1)[-1]), reverse=True,
        )
        for candidate in legacy:
            if checkpoint_manifest_valid(candidate):
                return candidate
    raise FileNotFoundError(f"No valid training checkpoint found in {ckpt_dir}")


def validate_checkpoint(
    path: Path,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate checkpoint bytes, manifest, schema, and model/config compatibility."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"Checkpoint missing or truncated: {path}")
    mp = manifest_path(path)
    if not mp.is_file():
        raise RuntimeError(f"Checkpoint integrity manifest missing: {mp}")
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Checkpoint manifest is invalid: {mp}") from exc
    if int(meta.get("schema", -1)) < MANIFEST_SCHEMA or meta.get("kind") != "checkpoint":
        raise RuntimeError(f"Checkpoint manifest schema/kind invalid: {mp}")
    if int(meta.get("size", -1)) != path.stat().st_size or meta.get("sha256") != sha256_file(path):
        raise RuntimeError(f"Checkpoint integrity verification failed: {path}")

    try:
        import torch
        from pipeline.trainer.model import LlamaModel, ModelConfig

        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to load checkpoint {path}: {exc}") from exc

    if not isinstance(payload, dict) or "model" not in payload or "model_cfg" not in payload or "step" not in payload:
        raise RuntimeError(f"Checkpoint schema invalid: {path}")
    try:
        cfg = ModelConfig.from_dict(payload["model_cfg"])
    except Exception as exc:
        raise RuntimeError(f"Checkpoint model config is invalid: {path}") from exc
    if expected_config is not None and cfg.to_dict() != expected_config:
        raise RuntimeError(f"Checkpoint model config mismatch: {path}")
    state = payload["model"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint model state is not a mapping: {path}")
    model = LlamaModel(cfg)
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint tensor shapes do not match its model config: {exc}") from exc
    missing = [x for x in missing if x not in {"rope_cos", "rope_sin"}]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint state is incompatible. Missing={missing}, unexpected={unexpected}")
    return payload


def atomic_jsonl_write(
    path: Path,
    producer: Callable[[], Iterable[Any]] | Iterable[Any],
) -> int:
    """Write an iterable or zero-argument producer atomically.

    Items may be ordinary JSON-serializable objects or already-serialized JSON
    records. Pre-serialized strings are validated before being written so JSONL
    callers cannot accidentally double-encode a complete JSON object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    count = 0
    try:
        items = producer() if callable(producer) else producer
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for item in items:
                if isinstance(item, str):
                    try:
                        json.loads(item)
                    except json.JSONDecodeError as exc:
                        raise ValueError("pre-serialized JSONL item is not valid JSON") from exc
                    line = item
                else:
                    line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                f.write(line + "\n")
                count += 1
        if count == 0:
            raise RuntimeError(f"Refusing to commit empty artifact: {path}")
        os.replace(tmp_name, path)
        return count
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
