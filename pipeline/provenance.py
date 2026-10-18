"""Configuration identity and artifact lineage helpers.

Phase 1 deliberately separates configuration identities.  The historical
pipeline_config_sha256 remains available as legacy evidence, but it is not
used as the identity of every downstream artifact.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


RUNTIME_PATH_KEYS = {"output_dir", "scratch_dir", "output_path", "shard_dir", "llamacpp_dir"}


def _semantic(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _semantic(item) for key, item in value.items() if key not in RUNTIME_PATH_KEYS}
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    return value


CONFIG_SECTIONS = {
    "dataset": ("crawl", "clean", "dedup", "weight"),
    "tokenizer": ("tokenizer",),
    "shard": ("shard",),
    "train": ("train",),
    "model": ("train",),
    "export": ("export",),
}

# These controls govern how a training run continues rather than what
# checkpoint state is compatible with that run. They must not invalidate
# deterministic checkpoint resume when intentionally changed.
TRAIN_CONTINUATION_KEYS = {
    "resume",
    "total_steps",
    "eval_every_steps",
    "eval_batches",
    "checkpoint_every_steps",
    "keep_checkpoints",
    "allow_cpu_training",
    "auto_size",
    "target_training_hours",
    "observed_tokens_per_sec",
}

def training_compatibility_config(value: Any) -> dict[str, Any]:
    """Return only training settings that define checkpoint compatibility."""
    train = dict(value or {})
    for key in TRAIN_CONTINUATION_KEYS:
        train.pop(key, None)
    return _semantic(train)


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def config_identities(cfg: dict[str, Any], *, pipeline_sha256: str | None = None,
                      source_definition_hashes: dict[str, str | None] | None = None) -> dict[str, Any]:
    """Return independent immutable configuration identities for a run."""
    identities: dict[str, Any] = {}
    for name, sections in CONFIG_SECTIONS.items():
        if name == "train":
            payload = {"train": training_compatibility_config(cfg.get("train", {}))}
        else:
            payload = {section: _semantic(cfg.get(section, {})) for section in sections}
        if name == "model":
            train = dict(payload.get("train") or {})
            payload["train"] = {
                key: train.get(key)
                for key in ("model_preset", "vocab_size", "seq_len", "dropout")
                if key in train
            }
        identities[f"{name}_config_sha256"] = stable_hash(payload)

    identities["source_definition_sha256"] = stable_hash(source_definition_hashes or {})
    if pipeline_sha256:
        identities["pipeline_config_sha256"] = pipeline_sha256
    return identities


def new_run_id() -> str:
    return uuid.uuid4().hex


def artifact_id(kind: str, content_sha256: str) -> str:
    return f"{kind}:{content_sha256}"


def snapshot_configs(root: Path, cfg: dict[str, Any], identities: dict[str, Any],
                     *, config_path: Path | None = None) -> dict[str, str]:
    """Persist immutable JSON snapshots keyed by their content identity."""
    directory = root / "provenance" / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    snapshots: dict[str, str] = {}

    for name, sections in CONFIG_SECTIONS.items():
        sha = identities[f"{name}_config_sha256"]
        payload = {section: _semantic(cfg.get(section, {})) for section in sections}
        if name == "model":
            train = dict(payload.get("train") or {})
            payload["train"] = {
                key: train.get(key)
                for key in ("model_preset", "vocab_size", "seq_len", "dropout")
                if key in train
            }
        path = directory / f"{name}-{sha}.json"
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        snapshots[name] = str(path.relative_to(root)).replace("\\", "/")

    if config_path is not None and config_path.is_file():
        legacy_sha = identities.get("pipeline_config_sha256")
        if legacy_sha:
            path = directory / f"pipeline-legacy-{legacy_sha}.yaml"
            if not path.exists():
                path.write_bytes(config_path.read_bytes())
            snapshots["pipeline_legacy"] = str(path.relative_to(root)).replace("\\", "/")
    return snapshots
