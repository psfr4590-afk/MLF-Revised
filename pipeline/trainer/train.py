"""Pretraining loop with AMP, gradient accumulation, evaluation and checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
try:
    import torch
    import torch.nn as nn
    _no_grad = torch.no_grad
except ImportError:  # Allow provenance and checkpoint manifest utilities to import without PyTorch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    def _no_grad():
        return lambda fn: fn

from pipeline.integrity import sha256_file
from pipeline.numerics import NonFiniteLossError, require_finite
from pipeline.model_sizer import estimate_total_tokens, profile_hardware, recommend_training_profile
from pipeline.preflight import discover_capabilities, estimate_duration, run_preflight, write_preflight_report
from pipeline.shardwriter.shard_writer import ShardDataLoader
from pipeline.provenance import (
    config_identities,
    artifact_id,
    stable_hash,
    training_compatibility_config,
)
from pipeline.trainer.model import LlamaModel, ModelConfig

log = logging.getLogger("trainer")


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shard_manifest_hash(shard_dir: Path) -> Optional[str]:
    manifest = shard_dir / "shards.manifest.json"
    return sha256_file(manifest) if manifest.is_file() else None


def _source_manifest_hash(cfg: dict) -> Optional[str]:
    output_dir = Path(cfg.get("pipeline", {}).get("output_dir", "output")).resolve()
    manifest = output_dir.parent / "source_manifest.json"
    return sha256_file(manifest) if manifest.is_file() else None


def _provenance(cfg: dict, model_cfg: ModelConfig, shard_dir: Path, train_cfg: Optional[dict] = None) -> dict:
    """Build checkpoint provenance without treating the pipeline config as universal identity."""
    source_sha = _source_manifest_hash(cfg)
    if not source_sha:
        raise RuntimeError("Training provenance requires the completed source manifest")
    shard_sha = _shard_manifest_hash(shard_dir)
    if not shard_sha:
        raise RuntimeError(f"Training provenance requires a valid shard manifest: {shard_dir}")

    effective_train_cfg = dict(train_cfg if train_cfg is not None else cfg.get("train", {}))
    effective_train_cfg["_source_manifest_sha256"] = source_sha
    identities = dict(cfg.get("_config_identities") or {})
    if not identities:
        identities = config_identities(
            cfg,
            pipeline_sha256=cfg.get("_pipeline_config_sha256"),
            source_definition_hashes=cfg.get("_source_definition_hashes") or {},
        )
    model_sha = stable_hash(model_cfg.to_dict())
    train_identity_cfg = training_compatibility_config(effective_train_cfg)
    train_sha = stable_hash(train_identity_cfg)
    identities["train_config_sha256"] = train_sha
    identities["model_config_sha256"] = model_sha
    shard_manifest = Path(shard_dir) / "shards.manifest.json"
    shard_manifest_artifact_id = artifact_id("shard-manifest", sha256_file(shard_manifest))
    tokenizer_manifest = Path(shard_dir).parent / "tokenizer" / "tokenizer.json.manifest.json"
    weighted_manifest = Path(shard_dir).parent.parent / "scratch" / "04_weighted.jsonl.manifest.json"
    source_manifest = Path(cfg.get("_source_manifest_path") or Path(shard_dir).parent.parent / "source_manifest.json")
    tokenizer_artifact_id = artifact_id("tokenizer-manifest", sha256_file(tokenizer_manifest)) if tokenizer_manifest.is_file() else None
    dataset_artifact_id = artifact_id("dataset-manifest", sha256_file(weighted_manifest)) if weighted_manifest.is_file() else None
    source_artifact_id = artifact_id("source-manifest", sha256_file(source_manifest)) if source_manifest.is_file() else None

    provenance = {
        "schema": 3,
        "run_id": cfg.get("_run_id"),
        "config_identities": identities,
        "config_paths": {"pipeline": str(cfg.get("_pipeline_config_relative_path") or cfg.get("_pipeline_config_path") or "").replace("\\", "/")},
        "config_snapshots": dict(cfg.get("_config_snapshots") or {}),
        "train_config_sha256": train_sha,
        "model_config_sha256": model_sha,
        "shard_manifest_sha256": shard_sha,
        "source_manifest_sha256": source_sha,
        "seed": int(effective_train_cfg.get("seed", 42)),
        "parent_artifact_ids": [x for x in (shard_manifest_artifact_id, tokenizer_artifact_id, dataset_artifact_id, source_artifact_id) if x],
        "tokenizer_artifact_id": tokenizer_artifact_id,
        "dataset_artifact_id": dataset_artifact_id,
        "source_artifact_id": source_artifact_id,
    }
    if cfg.get("_pipeline_config_sha256"):
        provenance["pipeline_config_sha256"] = cfg["_pipeline_config_sha256"]
    return provenance


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "state": [int(value) for value in numpy_state[1].tolist()],
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu() if torch is not None else None,
    }
    if torch is not None and torch.cuda.is_available():
        state["torch_cuda"] = [rng.cpu() for rng in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: dict) -> None:
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint RNG state is invalid")

    random.setstate(state["python"])

    numpy_state = state["numpy"]
    if isinstance(numpy_state, dict):
        np.random.set_state(
            (
                str(numpy_state["algorithm"]),
                np.asarray(numpy_state["state"], dtype=np.uint32),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    else:
        np.random.set_state(numpy_state)

    torch_cpu_state = state.get("torch_cpu", state.get("torch"))
    if torch_cpu_state is None:
        raise RuntimeError("Checkpoint CPU RNG state is missing")
    torch.set_rng_state(torch_cpu_state.cpu())

    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [rng.cpu() for rng in state["torch_cuda"]]
        )


def _rng_state_sha256(state: dict) -> str:
    return hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()


def _model_parameter_count(model: LlamaModel) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def cosine_lr(step: int, warmup_steps: int, lr_max: float, lr_min: float, total_steps: int) -> float:
    if step < warmup_steps: return lr_max * step / max(warmup_steps, 1)
    if step >= total_steps: return lr_min
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def save_checkpoint(
    model: LlamaModel,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    val_loss: Optional[float],
    cfg: dict,
    out_dir: Path,
    tag: str = "",
    provenance: Optional[dict] = None,
    train_loader: Optional[ShardDataLoader] = None,
    val_loader: Optional[ShardDataLoader] = None,
    training_metadata: Optional[dict] = None,
    best_step: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    final_step: Optional[int] = None,
):
    """Save an atomically replaced checkpoint and its integrity manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"ckpt_{tag}.pt" if tag in {"best", "final"} else f"ckpt_{step:07d}.pt"
    path = out_dir / name
    rng_state = _rng_state()
    metadata = dict(training_metadata or {})
    metadata["random_state"] = "payload.rng_state"
    metadata["random_state_sha256"] = _rng_state_sha256(rng_state)
    metadata["checkpoint_step"] = int(step)
    metadata["best_step"] = int(best_step) if best_step is not None else None
    metadata["best_val_loss"] = float(best_val_loss) if best_val_loss is not None else None
    metadata["final_step"] = int(final_step) if final_step is not None else None
    started = time.perf_counter()
    payload = {
        "schema": 4,
        "checkpoint_kind": tag or "periodic",
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "best_step": int(best_step) if best_step is not None else None,
        "final_step": int(final_step) if final_step is not None else None,
        "model_cfg": model.cfg.to_dict(),
        "train_cfg": cfg,
        "provenance": provenance or {},
        "training_metadata": metadata,
        "rng_state": rng_state,
        "loader_state": {
            "train": train_loader.state_dict() if train_loader is not None else None,
            "val": val_loader.state_dict() if val_loader is not None else None,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    checkpoint_duration = time.perf_counter() - started
    manifest = {
        "schema": 3,
        "kind": "checkpoint",
        "checkpoint_kind": tag or "periodic",
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "step": int(step),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "best_step": int(best_step) if best_step is not None else None,
        "final_step": int(final_step) if final_step is not None else None,
        "checkpoint_duration_seconds": checkpoint_duration,
        "provenance": provenance or {},
    }
    mp = path.with_name(path.name + ".manifest.json")
    mtmp = mp.with_suffix(mp.suffix + ".tmp")
    mtmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(mtmp, mp)
    return path


def _safe_torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> dict:
    """Load a checkpoint using PyTorch's restricted weights-only deserializer.

    Checkpoint hashes detect accidental or untracked changes, but a neighboring
    manifest is not a trust boundary: an attacker who can replace both files
    can otherwise supply a pickle payload. Never fall back to unrestricted
    pickle loading when safe deserialization is unavailable.
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe checkpoint loading requires a PyTorch version that supports "
            "weights_only=True; upgrade PyTorch before loading this checkpoint."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint schema invalid: {path}")
    return payload


def load_checkpoint(path: Path, model: LlamaModel, optimizer: Optional[torch.optim.Optimizer] = None, scaler=None, device: torch.device | None = None, expected_provenance: Optional[dict] = None, train_loader: Optional[ShardDataLoader] = None, val_loader: Optional[ShardDataLoader] = None) -> int:
    if not path.exists() or path.stat().st_size < 1024: raise RuntimeError(f"Checkpoint missing or truncated: {path}")
    mp = path.with_name(path.name + ".manifest.json")
    if not mp.is_file(): raise RuntimeError(f"Checkpoint integrity manifest missing: {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("kind") != "checkpoint" or int(meta.get("size", -1)) != path.stat().st_size or meta.get("sha256") != sha256_file(path): raise RuntimeError(f"Checkpoint integrity verification failed: {path}")
    if expected_provenance is not None and meta.get("provenance") != expected_provenance: raise RuntimeError(f"Checkpoint provenance mismatch: {path}. The checkpoint was produced from a different pipeline configuration, training configuration, model configuration, seed, shard manifest, or source manifest.")
    ckpt = _safe_torch_load(path, map_location=device or "cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt or "step" not in ckpt: raise RuntimeError(f"Checkpoint schema invalid: {path}")
    if int(ckpt.get("schema", 0)) < 4: raise RuntimeError(f"Checkpoint schema too old for deterministic resume: {path}")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        if "optimizer" not in ckpt: raise RuntimeError("Checkpoint has no optimizer state; refusing resume")
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None:
        if "scaler" not in ckpt: raise RuntimeError("Checkpoint has no scaler state; refusing resume")
        scaler.load_state_dict(ckpt["scaler"])
    loader_state = ckpt.get("loader_state") or {}
    if train_loader is not None:
        if not loader_state.get("train"): raise RuntimeError("Checkpoint has no deterministic train loader state; refusing resume")
        train_loader.load_state_dict(loader_state["train"])
    if val_loader is not None:
        if not loader_state.get("val"): raise RuntimeError("Checkpoint has no deterministic validation loader state; refusing resume")
        val_loader.load_state_dict(loader_state["val"])
    if "rng_state" not in ckpt: raise RuntimeError("Checkpoint has no deterministic RNG state; refusing resume")
    _restore_rng_state(ckpt["rng_state"])
    return int(ckpt["step"])


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    numbered = []
    for path in ckpt_dir.glob("ckpt_*.pt"):
        stem = path.stem
        if not stem.startswith("ckpt_") or not stem[5:].isdigit():
            continue
        if not path.with_name(path.name + ".manifest.json").is_file():
            continue
        numbered.append((int(stem[5:]), path))

    numbered.sort(key=lambda item: item[0], reverse=True)

    for step, path in numbered:
        try:
            checkpoint_metadata(path)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            log.warning(
                "Skipping invalid checkpoint during latest selection: %s (%s)",
                path,
                exc,
            )
            continue
        return path

    return None


def checkpoint_metadata(path: Path) -> dict:
    mp = path.with_name(path.name + ".manifest.json")
    if not mp.is_file(): raise RuntimeError(f"Checkpoint integrity manifest missing: {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("kind") != "checkpoint": raise RuntimeError(f"Invalid checkpoint manifest: {mp}")
    if int(meta.get("size", -1)) != path.stat().st_size or meta.get("sha256") != sha256_file(path):
        raise RuntimeError(f"Checkpoint integrity verification failed: {path}")
    return meta


def best_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    path = ckpt_dir / "ckpt_best.pt"
    if not path.is_file() or not path.with_name(path.name + ".manifest.json").is_file(): return None
    meta = checkpoint_metadata(path)
    return path if meta.get("checkpoint_kind") == "best" and meta.get("best_step") is not None else None


def select_checkpoint(ckpt_dir: Path, kind: str = "latest") -> Optional[Path]:
    if kind == "latest": return latest_checkpoint(ckpt_dir)
    if kind == "best": return best_checkpoint(ckpt_dir)
    if kind == "final":
        path = ckpt_dir / "ckpt_final.pt"
        return path if path.is_file() and path.with_name(path.name + ".manifest.json").is_file() else None
    raise ValueError(f"Unknown checkpoint selection kind: {kind!r}")


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg; self.t_cfg = cfg.get("train", {}); self.out_dir = Path(cfg.get("pipeline", {}).get("output_dir", "output")); self.ckpt_dir = self.out_dir / "checkpoints"; self.log_path = self.out_dir / "logs" / "train.log"; self.log_path.parent.mkdir(parents=True, exist_ok=True)
        target = str(self.log_path.resolve())
        self._log_handler = None
        if not any(getattr(h, "_model_lab_path", None) == target for h in log.handlers):
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler._model_lab_path = target
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            log.addHandler(handler)
            self._log_handler = handler

    def close(self) -> None:
        """Release the logging handler owned by this Trainer instance."""
        handler = self._log_handler
        self._log_handler = None
        if handler is not None:
            try:
                handler.flush()
            finally:
                log.removeHandler(handler)
                handler.close()

    def run(self):
        t = dict(self.t_cfg); seed = int(t.get("seed", 42)); _seed_everything(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else type("_Dev", (), {"type": "cpu"})()
        if device.type == "cpu" and not bool(t.get("allow_cpu_training", False)): raise RuntimeError("CUDA is unavailable and allow_cpu_training=false; refusing accidental CPU pretraining")
        requested_configuration = dict(t)
        hardware = profile_hardware()
        self._hardware_profile = hardware.to_dict()
        self._requested_configuration = requested_configuration
        self._initial_estimate_seconds = 0.0
        shard_dir = Path(t.get("shard_dir", self.out_dir / "shards"))
        if bool(t.get("auto_size", False)):
            try:
                total_tokens = estimate_total_tokens(shard_dir)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                total_tokens = None
            profile = recommend_training_profile(
                hardware,
                total_tokens=total_tokens,
                configured_steps=int(t.get("total_steps", 100_000)),
                target_training_hours=float(t["target_training_hours"]) if t.get("target_training_hours") is not None else None,
                observed_tokens_per_sec=float(t["observed_tokens_per_sec"]) if t.get("observed_tokens_per_sec") is not None else None,
                max_seq_len=(
                    int(self.cfg["shard"]["sequence_length"])
                    if self.cfg.get("shard", {}).get("sequence_length") is not None
                    else None
                ),
            )
            t.update({
                "model_preset": profile.model_preset,
                "seq_len": profile.seq_len,
                "batch_size": profile.batch_size,
                "grad_accum_steps": profile.grad_accum_steps,
                "eval_batches": profile.eval_batches,
                "eval_every_steps": profile.eval_every_steps,
                "checkpoint_every_steps": profile.checkpoint_every_steps,
                "total_steps": profile.recommended_steps or int(t.get("total_steps", 100_000)),
                "_hardware_profile": hardware.to_dict(),
                "_training_profile": profile.to_dict(),
            })
            log.info("Auto-sized training profile | hardware=%s | profile=%s", hardware.to_dict(), profile.to_dict())
        effective_configuration = dict(t)
        preflight_cfg = self.cfg.get("preflight", {})
        preflight_enabled = bool(preflight_cfg.get("enabled", True))
        preflight_result = None
        estimate = None
        if preflight_enabled:
            preflight_result = run_preflight(
                t,
                shard_dir,
                warmup_steps=int(preflight_cfg.get("warmup_steps", 1)),
                benchmark_steps=int(preflight_cfg.get("benchmark_steps", 3)),
            )
            if not preflight_result.viable:
                raise RuntimeError(f"Preflight rejected training configuration: {preflight_result.failure}")
            estimate = estimate_duration(
                total_steps=int(t.get("total_steps", 100_000)),
                batch_size=int(t.get("batch_size", 1)),
                grad_accum_steps=int(t.get("grad_accum_steps", 1)),
                seq_len=int(t.get("seq_len", 256)),
                tokens_per_sec=preflight_result.tokens_per_sec,
                eval_every_steps=int(t.get("eval_every_steps", 500)),
                eval_batches=int(t.get("eval_batches", 20)),
                checkpoint_every_steps=int(t.get("checkpoint_every_steps", 1000)),
                eval_seconds=(
                    float(t["estimated_eval_seconds"])
                    if t.get("estimated_eval_seconds") is not None
                    else preflight_result.eval_seconds_per_event
                ),
                checkpoint_seconds=(
                    float(t["estimated_checkpoint_seconds"])
                    if t.get("estimated_checkpoint_seconds") is not None
                    else preflight_result.checkpoint_seconds_per_event
                ),
            )
            capability_report = None
            if bool(preflight_cfg.get("capability_discovery", False)):
                capability_report = discover_capabilities(
                    hardware,
                    benchmark_steps=int(preflight_cfg.get("capability_benchmark_steps", 2)),
                    max_model_params=int(preflight_cfg.get("capability_max_model_params", 1_000_000_000)),
                    max_context=int(preflight_cfg.get("capability_max_context", 4096)),
                    model_probe_rounds=int(preflight_cfg.get("capability_model_probe_rounds", 4)),
                    context_probe_rounds=int(preflight_cfg.get("capability_context_probe_rounds", 5)),
                    vocab_size=int(t.get("vocab_size", 32000)),
                )
            write_preflight_report(self.out_dir / "preflight_report.json", {
                "hardware": hardware.to_dict(),
                "requested_configuration": requested_configuration,
                "effective_configuration": effective_configuration,
                "decision_reasons": list(t.get("_training_profile", {}).get("decision_reasons", [])),
                "benchmark": preflight_result.to_dict(),
                "estimate": estimate,
                "capability_discovery": capability_report,
            })
            t["_preflight"] = preflight_result.to_dict()
            t["_initial_estimate"] = estimate
        preset = t.get("model_preset", "117M"); model_cfg = ModelConfig.from_preset(preset); model_cfg.vocab_size = int(t.get("vocab_size", 32000)); model_cfg.seq_len = int(t.get("seq_len", 1024)); model_cfg.dropout = float(t.get("dropout", 0.0)); model = LlamaModel(model_cfg).to(device)
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if p.requires_grad: (no_decay if (p.dim() < 2 or "norm" in name or "bias" in name or "embed" in name) else decay).append(p)
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": float(t.get("weight_decay", 0.1))}, {"params": no_decay, "weight_decay": 0.0}], lr=float(t.get("lr_max", 3e-4)), betas=(0.9, 0.95), fused=False)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda")); self.amp_dtype = torch.float16
        total_steps = int(t.get("total_steps", 100_000)); warmup_steps = int(t.get("warmup_steps", 2_000)); batch_size = int(t.get("batch_size", 1)); grad_accum = int(t.get("grad_accum_steps", 16)); grad_clip = float(t.get("grad_clip", 1.0)); eval_every = max(1, int(t.get("eval_every_steps", 500))); ckpt_every = max(1, int(t.get("checkpoint_every_steps", 1000))); keep_checkpoints = max(1, int(t.get("keep_checkpoints", 3))); eval_batches = max(1, int(t.get("eval_batches", 20))); seq_len = model_cfg.seq_len; shard_dtype = str(t.get("shard_dtype", self.cfg.get("shard", {}).get("dtype", "uint16"))); dtype = np.uint32 if shard_dtype == "uint32" else np.uint16; shard_dir = Path(t.get("shard_dir", self.out_dir / "shards"))
        if max_seq_len := self.cfg.get("shard", {}).get("sequence_length"):
            if seq_len > int(max_seq_len): raise RuntimeError(f"Training seq_len={seq_len} exceeds shard sequence_length={int(max_seq_len)}; rebuild shards or enable a compatible auto-size profile")
        train_loader = ShardDataLoader(shard_dir, "train", seq_len, dtype=dtype, seed=seed); val_loader = ShardDataLoader(shard_dir, "val", seq_len, dtype=dtype, seed=seed); provenance = _provenance(self.cfg, model_cfg, shard_dir, train_cfg=t)
        start_step, best_val, best_step = 0, float("inf"), None
        tokens_processed = 0
        evaluation_duration = 0.0
        checkpoint_duration = 0.0
        if bool(t.get("resume", True)):
            ckpt = latest_checkpoint(self.ckpt_dir)
            if ckpt:
                start_step = load_checkpoint(
                    ckpt, model, optimizer, scaler, device,
                    expected_provenance=provenance,
                    train_loader=train_loader, val_loader=val_loader,
                )
                payload = _safe_torch_load(ckpt, map_location="cpu")
                best_val = float(payload.get("best_val_loss", payload.get("val_loss", best_val)))
                best_step = payload.get("best_step")
                saved_meta = payload.get("training_metadata", {})
                tokens_processed = int(saved_meta.get(
                    "tokens_processed", start_step * batch_size * grad_accum * seq_len
                ))
                evaluation_duration = float(saved_meta.get("evaluation_duration_seconds", 0.0))
                checkpoint_duration = float(saved_meta.get("checkpoint_duration_seconds", 0.0))
        metrics_path = self.out_dir / "logs" / "metrics.jsonl"
        metrics = open(metrics_path, "a", encoding="utf-8")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        initial_estimate_seconds = float((estimate or {}).get("estimated_duration_seconds") or 0.0)
        self._initial_estimate_seconds = initial_estimate_seconds
        step = start_step
        try:
            while step < total_steps:
                lr = cosine_lr(step, warmup_steps, float(t.get("lr_max", 3e-4)), float(t.get("lr_min", 3e-5)), total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                accum_loss = 0.0
                for _ in range(grad_accum):
                    x, y = train_loader.next_batch(batch_size)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=(device.type == "cuda")):
                        _, loss = model(x, y)
                        loss = loss / grad_accum
                    micro_loss = require_finite("train_loss", loss.item(), step + 1)
                    scaler.scale(loss).backward()
                    accum_loss += micro_loss
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                tokens_processed += batch_size * grad_accum * seq_len
                if step % 10 == 0:
                    elapsed = max(time.time() - start_time, 1e-6)
                    tok_s = (step - start_step) * batch_size * grad_accum * seq_len / elapsed
                    remaining_tokens = max(0, total_steps - step) * batch_size * grad_accum * seq_len
                    remaining_seconds = remaining_tokens / tok_s if tok_s > 0 else None
                    live_estimate_seconds = elapsed + remaining_seconds if remaining_seconds is not None else None
                    metrics.write(json.dumps({
                        "step": step, "train_loss": accum_loss, "lr": lr,
                        "grad_norm": float(grad_norm), "tokens_per_sec": tok_s,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": remaining_seconds,
                        "live_estimate_seconds": live_estimate_seconds,
                        "initial_estimate_seconds": initial_estimate_seconds,
                        "actual_seconds_so_far": elapsed,
                    }) + "\n")
                    metrics.flush()
                if step % eval_every == 0:
                    eval_started = time.perf_counter()
                    val_loss = self._eval(model, val_loader, device, eval_batches, batch_size)
                    evaluation_duration += time.perf_counter() - eval_started
                    metrics.write(json.dumps({
                        "step": step, "val_loss": val_loss,
                        "val_ppl": math.exp(min(val_loss, 20))
                    }) + "\n")
                    metrics.flush()
                    model.train()
                    if val_loss < best_val:
                        best_val = val_loss
                        best_step = step
                        save_started = time.perf_counter()
                        save_checkpoint(
                            model, optimizer, scaler, step, val_loss, t, self.ckpt_dir,
                            tag="best", provenance=provenance,
                            train_loader=train_loader, val_loader=val_loader,
                            training_metadata=self._training_metadata(
                                model, t, seed, step, total_steps, batch_size, grad_accum,
                                seq_len, tokens_processed, start_time,
                                evaluation_duration, checkpoint_duration
                            ),
                            best_step=best_step, best_val_loss=best_val,
                        )
                        checkpoint_duration += time.perf_counter() - save_started
                if step % ckpt_every == 0:
                    save_started = time.perf_counter()
                    save_checkpoint(
                        model, optimizer, scaler, step, None, t, self.ckpt_dir,
                        provenance=provenance, train_loader=train_loader, val_loader=val_loader,
                        training_metadata=self._training_metadata(
                            model, t, seed, step, total_steps, batch_size, grad_accum,
                            seq_len, tokens_processed, start_time,
                            evaluation_duration, checkpoint_duration
                        ),
                        best_step=best_step, best_val_loss=best_val,
                    )
                    checkpoint_duration += time.perf_counter() - save_started
                    self._prune_checkpoints(self.ckpt_dir, keep=keep_checkpoints)
            final_eval_started = time.perf_counter()
            final_val_loss = self._eval(model, val_loader, device, eval_batches, batch_size)
            evaluation_duration += time.perf_counter() - final_eval_started
            save_started = time.perf_counter()
            save_checkpoint(
                model, optimizer, scaler, step, final_val_loss, t, self.ckpt_dir,
                tag="final", provenance=provenance,
                train_loader=train_loader, val_loader=val_loader,
                training_metadata=self._training_metadata(
                    model, t, seed, step, total_steps, batch_size, grad_accum,
                    seq_len, tokens_processed, start_time,
                    evaluation_duration, checkpoint_duration
                ),
                best_step=best_step, best_val_loss=best_val, final_step=step,
            )
            checkpoint_duration += time.perf_counter() - save_started
            return model, step
        finally:
            metrics.close()

    def _training_metadata(
        self, model, t, seed, step, total_steps, batch_size, grad_accum,
        seq_len, tokens_processed, start_time, evaluation_duration, checkpoint_duration
    ) -> dict:
        return {
            "model_parameters": _model_parameter_count(model),
            "architecture": str(t.get("model_preset", "117M")),
            "sequence_length": int(seq_len),
            "batch_size": int(batch_size),
            "effective_batch_size": int(batch_size * grad_accum),
            "gradient_accumulation": int(grad_accum),
            "optimizer": "AdamW",
            "lr": float(t.get("lr_max", 3e-4)),
            "scheduler": "cosine",
            "precision": "float16" if (torch is not None and torch.cuda.is_available()) else "float32",
            "seed": int(seed),
            "total_steps": int(total_steps),
            "step": int(step),
            "tokens_processed": int(tokens_processed),
            "training_duration_seconds": max(0.0, time.time() - start_time),
            "evaluation_duration_seconds": float(evaluation_duration),
            "checkpoint_duration_seconds": float(checkpoint_duration),
            "hardware_profile": t.get("_hardware_profile", self._hardware_profile),
            "requested_configuration": self._requested_configuration,
            "effective_configuration": {k: v for k, v in t.items() if not k.startswith("__")},
            "preflight": t.get("_preflight"),
            "initial_estimate_seconds": self._initial_estimate_seconds,
            "estimated_vs_actual_seconds": (
                {"estimated_seconds": self._initial_estimate_seconds, "actual_seconds": max(0.0, time.time() - start_time)}
                if self._initial_estimate_seconds > 0 else None
            ),
        }

    @_no_grad()
    def _eval(self, model, loader, device, n_batches, batch_size) -> float:
        model.eval(); losses = []
        for _ in range(n_batches):
            x, y = loader.next_batch(batch_size); x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=(device.type == "cuda")): _, loss = model(x, y)
            losses.append(require_finite("validation_loss", loss.item()))
        return require_finite("mean_validation_loss", float(np.mean(losses)))

    def _prune_checkpoints(self, ckpt_dir: Path, keep: int = 3):
        checkpoints = sorted(ckpt_dir.glob("ckpt_[0-9]*.pt"))
        for old in checkpoints[:-keep]: old.unlink(missing_ok=True); old.with_name(old.name + ".manifest.json").unlink(missing_ok=True)


__all__ = ["Trainer", "NonFiniteLossError", "require_finite", "cosine_lr", "save_checkpoint", "load_checkpoint", "checkpoint_metadata", "latest_checkpoint", "best_checkpoint", "select_checkpoint"]
