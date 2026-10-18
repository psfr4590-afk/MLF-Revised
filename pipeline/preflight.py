"""Preflight training benchmark and duration estimation.

The benchmark uses the real model geometry and real shard loader, but a fresh
model/optimizer, so it validates viability without mutating a training run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import hashlib
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from pipeline.model_sizer import HardwareProfile
from pipeline.shardwriter.shard_writer import ShardDataLoader
from pipeline.trainer.model import LlamaModel, ModelConfig


@dataclass(frozen=True)
class PreflightResult:
    viable: bool
    tokens_per_sec: float | None
    steps_per_sec: float | None
    peak_memory_gb: float | None
    gpu_utilization_percent: float | None
    benchmark_seconds: float
    warmup_steps: int
    benchmark_steps: int
    failure: str | None = None
    eval_seconds_per_event: float | None = None
    checkpoint_seconds_per_event: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gpu_utilization() -> float | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return float(out.stdout.splitlines()[0].strip())
    except ValueError:
        return None


def _make_model(train_cfg: dict[str, Any]) -> LlamaModel:
    cfg = ModelConfig.from_preset(str(train_cfg.get("model_preset", "85M")))
    cfg.vocab_size = int(train_cfg.get("vocab_size", 32000))
    cfg.seq_len = int(train_cfg.get("seq_len", 256))
    cfg.dropout = float(train_cfg.get("dropout", 0.0))
    return LlamaModel(cfg)



def _benchmark_eval_seconds(
    train_cfg: dict[str, Any],
    shard_dir: str | Path,
    model: LlamaModel,
    device: torch.device,
    *,
    batch_size: int,
    seq_len: int,
    dtype: Any,
) -> float:
    """Measure one configured validation pass on real validation shards."""
    batches = max(1, int(train_cfg.get("eval_batches", 20)))
    loader = ShardDataLoader(
        shard_dir, "val", seq_len, dtype=dtype, seed=int(train_cfg.get("seed", 42))
    )
    model.eval()
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for _ in range(batches):
                x, y = loader.next_batch(batch_size)
                x, y = x.to(device), y.to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    model(x, y)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        return max(0.0, time.perf_counter() - started)
    finally:
        model.train()


def _benchmark_checkpoint_seconds(
    model: LlamaModel,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    shard_dir: str | Path,
) -> float:
    """Measure checkpoint serialization, disk write, hashing, and manifest write."""
    checkpoint_path: Path | None = None
    manifest_path: Path | None = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".mlframework-preflight-", suffix=".pt",
            dir=shard_dir, delete=False,
        ) as handle:
            checkpoint_path = Path(handle.name)
        manifest_path = checkpoint_path.with_name(checkpoint_path.name + ".manifest.json")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, checkpoint_path)
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest_path.write_text(json.dumps({
            "kind": "preflight-checkpoint",
            "size": checkpoint_path.stat().st_size,
            "sha256": digest.hexdigest(),
        }, sort_keys=True), encoding="utf-8")
        return max(0.0, time.perf_counter() - started)
    finally:
        if checkpoint_path is not None:
            checkpoint_path.unlink(missing_ok=True)
        if manifest_path is not None:
            manifest_path.unlink(missing_ok=True)


def run_preflight(
    train_cfg: dict[str, Any],
    shard_dir: str | Path,
    *,
    warmup_steps: int = 1,
    benchmark_steps: int = 3,
) -> PreflightResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else type("_Dev", (), {"type": "cpu"})()
    warmup_steps = max(0, int(warmup_steps))
    benchmark_steps = max(1, int(benchmark_steps))
    started = time.perf_counter()
    model = None
    optimizer = None
    try:
        model = _make_model(train_cfg).to(device)
        batch_size = max(1, int(train_cfg.get("batch_size", 1)))
        grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        seq_len = int(train_cfg.get("seq_len", model.cfg.seq_len))
        dtype_name = str(train_cfg.get("shard_dtype", "uint16"))
        import numpy as np
        dtype = np.uint32 if dtype_name == "uint32" else np.uint16
        loader = ShardDataLoader(shard_dir, "train", seq_len, dtype=dtype, seed=int(train_cfg.get("seed", 42)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr_max", 3e-4)), weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model.train()

        def step_once(model=model, optimizer=optimizer) -> None:
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                x, y = loader.next_batch(batch_size)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    _, loss = model(x, y)
                    loss = loss / grad_accum
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        for _ in range(warmup_steps):
            step_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        started_bench = time.perf_counter()
        for _ in range(benchmark_steps):
            step_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started_bench, 1e-9)
        tokens = benchmark_steps * batch_size * grad_accum * seq_len
        tok_s = tokens / elapsed

        # Sample utilization after timing. Launching nvidia-smi inside the
        # measured interval would contaminate throughput, especially for short
        # benchmarks.
        util = _gpu_utilization()
        peak = (torch.cuda.max_memory_allocated() / (1024**3)) if device.type == "cuda" else None
        eval_seconds = None
        checkpoint_seconds = None
        try:
            eval_seconds = _benchmark_eval_seconds(
                train_cfg, shard_dir, model, device,
                batch_size=batch_size, seq_len=seq_len, dtype=dtype,
            )
        except (RuntimeError, OSError, ValueError, TypeError):
            # Throughput viability is independent of whether overhead calibration succeeds.
            pass
        try:
            checkpoint_seconds = _benchmark_checkpoint_seconds(
                model, optimizer, scaler, shard_dir,
            )
        except (RuntimeError, OSError, ValueError, TypeError):
            # Report missing checkpoint calibration rather than rejecting a viable model.
            pass
        return PreflightResult(
            viable=True,
            tokens_per_sec=tok_s,
            steps_per_sec=benchmark_steps / elapsed,
            peak_memory_gb=peak,
            gpu_utilization_percent=util,
            benchmark_seconds=time.perf_counter() - started,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
            eval_seconds_per_event=eval_seconds,
            checkpoint_seconds_per_event=checkpoint_seconds,
        )
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        return PreflightResult(
            viable=False, tokens_per_sec=None, steps_per_sec=None,
            peak_memory_gb=(torch.cuda.max_memory_allocated() / (1024**3)) if device.type == "cuda" else None,
            gpu_utilization_percent=None, benchmark_seconds=time.perf_counter() - started,
            warmup_steps=warmup_steps, benchmark_steps=benchmark_steps,
            failure=f"{type(exc).__name__}: {exc}",
        )
    finally:
        del optimizer
        del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def discover_capabilities(
    hardware: HardwareProfile,
    *,
    candidates: tuple[tuple[str, int], ...] | None = None,
    benchmark_steps: int = 2,
    vocab_size: int = 32000,
    max_model_params: int = 1_000_000_000,
    max_context: int = 4096,
    model_probe_rounds: int = 4,
    context_probe_rounds: int = 5,
) -> dict[str, Any]:
    """Empirically map the usable model-size/context boundary.

    When explicit candidates are supplied, they are probed exactly for
    backwards-compatible tests and targeted diagnostics. Otherwise discovery
    starts with coarse parameter-budget probes, constructs real intermediate
    architectures, then refines the last-success/first-failure interval.
    Context is probed independently on the largest verified model.

    Synthetic tokens isolate hardware capability from shard geometry. The
    normal run_preflight() benchmark remains the production-shard viability
    gate.
    """
    results: list[dict[str, Any]] = []
    steps = max(1, int(benchmark_steps))
    device = torch.device("cuda" if hardware.cuda_available else "cpu")

    def probe(cfg: ModelConfig, *, source: str, target_params: int | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        model = None
        optimizer = None
        try:
            model = LlamaModel(cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
            scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
            model.train()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            for _ in range(steps):
                x = torch.randint(0, int(cfg.vocab_size), (1, int(cfg.seq_len)), device=device)
                y = torch.randint(0, int(cfg.vocab_size), (1, int(cfg.seq_len)), device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - started, 1e-9)
            peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else None
            return {
                "source": source,
                "target_params": target_params,
                "model_params": int(cfg.param_count()),
                "architecture": {
                    "n_layers": int(cfg.n_layers),
                    "n_heads": int(cfg.n_heads),
                    "n_kv_heads": int(cfg.n_kv_heads),
                    "d_model": int(cfg.d_model),
                    "d_ffn": int(cfg.d_ffn),
                    "vocab_size": int(cfg.vocab_size),
                },
                "seq_len": int(cfg.seq_len),
                "batch_size": 1,
                "benchmark_steps": steps,
                "viable": True,
                "tokens_per_sec": (steps * int(cfg.seq_len)) / elapsed,
                "peak_memory_gb": peak,
                "elapsed_seconds": elapsed,
                "failure": None,
            }
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
            return {
                "source": source,
                "target_params": target_params,
                "model_params": int(cfg.param_count()),
                "architecture": {
                    "n_layers": int(cfg.n_layers),
                    "n_heads": int(cfg.n_heads),
                    "n_kv_heads": int(cfg.n_kv_heads),
                    "d_model": int(cfg.d_model),
                    "d_ffn": int(cfg.d_ffn),
                    "vocab_size": int(cfg.vocab_size),
                },
                "seq_len": int(cfg.seq_len),
                "batch_size": 1,
                "benchmark_steps": steps,
                "viable": False,
                "tokens_per_sec": None,
                "peak_memory_gb": (
                    torch.cuda.max_memory_allocated() / (1024**3)
                    if device.type == "cuda" else None
                ),
                "elapsed_seconds": max(time.perf_counter() - started, 0.0),
                "failure": f"{type(exc).__name__}: {exc}",
            }
        finally:
            del optimizer
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if candidates is not None:
        for model_preset, seq_len in candidates:
            cfg = ModelConfig.from_preset(str(model_preset))
            cfg.vocab_size = int(vocab_size)
            cfg.seq_len = int(seq_len)
            results.append(probe(cfg, source="explicit"))
    else:
        # Named presets remain anchor points, but intermediate architectures
        # are generated from the actual parameter budget instead of invented
        # preset names. A coarse pass finds the boundary, then binary
        # refinement resolves the useful region.
        coarse = [64_000_000, 85_000_000, 117_000_000, 180_000_000,
                  260_000_000, 360_000_000, 512_000_000]
        if device.type == "cuda":
            coarse.append(int(max_model_params))
        else:
            coarse = [p for p in coarse if p <= 117_000_000]
        coarse = sorted({p for p in coarse if 1_000_000 <= p <= int(max_model_params)})
        last_success: int | None = None
        first_failure: int | None = None
        tested_targets: set[int] = set()

        for target in coarse:
            if target in tested_targets:
                continue
            tested_targets.add(target)
            cfg = ModelConfig.from_target_params(target, seq_len=128, vocab_size=vocab_size)
            result = probe(cfg, source="coarse", target_params=target)
            results.append(result)
            if result["viable"]:
                last_success = target
            else:
                first_failure = target
                break

        for _ in range(max(0, int(model_probe_rounds))):
            if last_success is None or first_failure is None:
                break
            if first_failure - last_success < 1_000_000:
                break
            target = (last_success + first_failure) // 2
            if target in tested_targets:
                break
            tested_targets.add(target)
            cfg = ModelConfig.from_target_params(target, seq_len=128, vocab_size=vocab_size)
            result = probe(cfg, source="refinement", target_params=target)
            results.append(result)
            if result["viable"]:
                last_success = target
            else:
                first_failure = target

        viable = [item for item in results if item["viable"]]
        if viable:
            largest = max(viable, key=lambda item: int(item["model_params"]))
            base_cfg = ModelConfig.from_dict({
                **largest["architecture"],
                "seq_len": 128,
            })
            context = 128
            context_success = 128
            for _ in range(max(0, int(context_probe_rounds))):
                next_context = min(int(max_context), context * 2)
                if next_context <= context:
                    break
                context = next_context
                cfg = ModelConfig.from_dict({**base_cfg.to_dict(), "seq_len": context})
                result = probe(cfg, source="context", target_params=largest.get("target_params"))
                results.append(result)
                if result["viable"]:
                    context_success = context
                else:
                    break
            if context_success < int(max_context):
                lower = context_success
                upper = context
                for _ in range(max(0, int(context_probe_rounds))):
                    if upper - lower <= 1:
                        break
                    mid = (lower + upper) // 2
                    cfg = ModelConfig.from_dict({**base_cfg.to_dict(), "seq_len": mid})
                    result = probe(cfg, source="context_refinement", target_params=largest.get("target_params"))
                    results.append(result)
                    if result["viable"]:
                        lower = mid
                    else:
                        upper = mid

    verified = [item for item in results if item["viable"]]
    failed = [item for item in results if not item["viable"]]
    return {
        "schema": 2,
        "hardware": hardware.to_dict(),
        "tested_combinations": len(results),
        "verified_combinations": len(verified),
        "failed_combinations": len(failed),
        "max_verified_model_params": max((int(item["model_params"]) for item in verified), default=None),
        "max_verified_context": max((int(item["seq_len"]) for item in verified), default=None),
        "results": results,
    }

def estimate_duration(
    *,
    total_steps: int,
    batch_size: int,
    grad_accum_steps: int,
    seq_len: int,
    tokens_per_sec: float | None,
    eval_every_steps: int,
    eval_batches: int,
    checkpoint_every_steps: int,
    checkpoint_seconds: float | None = None,
    eval_seconds: float | None = None,
) -> dict[str, Any]:
    if not tokens_per_sec or tokens_per_sec <= 0:
        return {"estimated_duration_seconds": None, "estimated_completion": None}

    total_steps = max(0, int(total_steps))
    batch_size = max(1, int(batch_size))
    grad_accum_steps = max(1, int(grad_accum_steps))
    seq_len = max(1, int(seq_len))
    eval_interval = max(1, int(eval_every_steps))
    checkpoint_interval = max(1, int(checkpoint_every_steps))
    missing_overhead_measurements = []
    if eval_seconds is None:
        missing_overhead_measurements.append("evaluation")
        eval_seconds_value = 0.0
    else:
        eval_seconds_value = max(0.0, float(eval_seconds))
    if checkpoint_seconds is None:
        missing_overhead_measurements.append("checkpoint")
        checkpoint_seconds_value = 0.0
    else:
        checkpoint_seconds_value = max(0.0, float(checkpoint_seconds))

    tokens_per_step = batch_size * grad_accum_steps * seq_len
    training_seconds = total_steps * tokens_per_step / tokens_per_sec
    scheduled_eval_count = total_steps // eval_interval
    scheduled_checkpoint_count = total_steps // checkpoint_interval
    # Trainer.run always performs a final evaluation and final checkpoint after
    # the loop. A best-checkpoint save is also attempted after every scheduled
    # evaluation, so count one checkpoint per evaluation as a conservative
    # allowance for improving validation loss.
    eval_count = scheduled_eval_count + 1
    checkpoint_count = scheduled_checkpoint_count + scheduled_eval_count + 1

    checkpoints = []
    evaluations = []
    elapsed_overhead = 0.0
    for step in range(1, total_steps + 1):
        # The trainer evaluates before best/scheduled checkpoints at each step.
        if step % eval_interval == 0:
            elapsed_overhead += eval_seconds_value
            evaluations.append({
                "step": step,
                "estimated_elapsed_seconds": round(
                    step * tokens_per_step / tokens_per_sec + elapsed_overhead,
                    3,
                ),
            })
            elapsed_overhead += checkpoint_seconds_value
            checkpoints.append({
                "step": step,
                "kind": "best-checkpoint allowance",
                "estimated_elapsed_seconds": round(
                    step * tokens_per_step / tokens_per_sec + elapsed_overhead,
                    3,
                ),
            })
        if step % checkpoint_interval == 0:
            elapsed_overhead += checkpoint_seconds_value
            checkpoints.append({
                "step": step,
                "kind": "scheduled",
                "estimated_elapsed_seconds": round(
                    step * tokens_per_step / tokens_per_sec + elapsed_overhead,
                    3,
                ),
            })

    # Final evaluation and final checkpoint always run after the loop, even
    # when the final step is also an evaluation/checkpoint interval.
    elapsed_overhead += eval_seconds_value
    evaluations.append({
        "step": total_steps,
        "kind": "final",
        "estimated_elapsed_seconds": round(
            training_seconds + elapsed_overhead, 3,
        ),
    })
    elapsed_overhead += checkpoint_seconds_value
    checkpoints.append({
        "step": total_steps,
        "kind": "final",
        "estimated_elapsed_seconds": round(
            training_seconds + elapsed_overhead, 3,
        ),
    })

    total_seconds = (
        training_seconds
        + eval_count * eval_seconds_value
        + checkpoint_count * checkpoint_seconds_value
    )
    return {
        "estimated_training_seconds": training_seconds,
        "estimated_duration_seconds": total_seconds,
        "estimated_eval_count": eval_count,
        "estimated_checkpoint_count": checkpoint_count,
        "overhead_estimate_complete": not missing_overhead_measurements,
        "missing_overhead_measurements": missing_overhead_measurements,
        "estimated_eval_seconds_per_event": eval_seconds_value,
        "estimated_checkpoint_seconds_per_event": checkpoint_seconds_value,
        "estimated_checkpoint_times": checkpoints,
        "estimated_eval_times": evaluations,
        "estimated_completion": time.time() + total_seconds,
    }

def write_preflight_report(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(target)


__all__ = ["PreflightResult", "run_preflight", "discover_capabilities", "estimate_duration", "write_preflight_report"]
