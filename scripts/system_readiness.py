#!/usr/bin/env python3
"""CLI for the unified System Readiness workflow."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "runtime" / "system_readiness.json"


def _benchmark_worker(root: Path, warmup_steps: int, benchmark_steps: int) -> dict[str, Any]:
    import yaml
    from pipeline.config_validation import validate_config
    from pipeline.preflight import run_preflight

    config_path = root / "config" / "pipeline_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    validate_config(config)
    train_cfg = dict(config.get("train", {}))
    shard_dir = root / str(config.get("shard", {}).get("output_dir", "output/shards"))
    if not shard_dir.is_absolute():
        shard_dir = root / shard_dir
    if not (shard_dir / "shards.manifest.json").is_file():
        raise FileNotFoundError(f"Verified shard manifest not found: {shard_dir / 'shards.manifest.json'}")
    train_cfg["shard_dtype"] = config.get("shard", {}).get("dtype", "uint16")
    result = run_preflight(train_cfg, shard_dir, warmup_steps=warmup_steps, benchmark_steps=benchmark_steps)
    return result.to_dict()


def _run_benchmark_subprocess(root: Path, warmup_steps: int, benchmark_steps: int, timeout: float) -> dict[str, Any]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--benchmark-worker",
        "--root", str(root), "--warmup-steps", str(warmup_steps),
        "--benchmark-steps", str(benchmark_steps),
    ]
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"viable": False, "failure": f"Benchmark exceeded its {timeout:.0f}-second wall-clock limit and was terminated.",
                "benchmark_seconds": timeout, "warmup_steps": warmup_steps, "benchmark_steps": benchmark_steps}
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or f"worker exited {completed.returncode}").strip()
        return {"viable": False, "failure": detail[-4000:], "benchmark_seconds": None,
                "warmup_steps": warmup_steps, "benchmark_steps": benchmark_steps}
    try:
        payload = json.loads(completed.stdout)
        return payload if isinstance(payload, dict) else {"viable": False, "failure": "Benchmark worker returned a non-object JSON result."}
    except json.JSONDecodeError:
        return {"viable": False, "failure": "Benchmark worker returned invalid JSON.", "benchmark_seconds": None,
                "warmup_steps": warmup_steps, "benchmark_steps": benchmark_steps}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether this Model Lab installation is ready to run safely.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root (defaults to this checkout).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Atomic JSON report path.")
    parser.add_argument("--benchmark", action="store_true", help="Run a bounded benchmark against existing verified training shards.")
    parser.add_argument("--benchmark-timeout", type=float, default=90.0, help="Hard wall-clock limit for the isolated benchmark subprocess.")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--benchmark-steps", type=int, default=2)
    parser.add_argument("--benchmark-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()

    if args.benchmark_worker:
        try:
            print(json.dumps(_benchmark_worker(root, max(0, args.warmup_steps), max(1, args.benchmark_steps)),
                             sort_keys=True, allow_nan=False))
            return 0
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    if args.benchmark_timeout <= 0:
        parser.error("--benchmark-timeout must be > 0")
    if args.warmup_steps < 0 or args.benchmark_steps < 1:
        parser.error("--warmup-steps must be >= 0 and --benchmark-steps must be >= 1")

    from pipeline.readiness import build_readiness_report, write_readiness_report

    benchmark = _run_benchmark_subprocess(root, args.warmup_steps, args.benchmark_steps,
                                          args.benchmark_timeout) if args.benchmark else None
    report = build_readiness_report(root, benchmark=benchmark)
    output = args.output if args.output.is_absolute() else root / args.output
    written = write_readiness_report(report, output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"Readiness report: {written}")
    return 0 if report["summary"]["required_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
