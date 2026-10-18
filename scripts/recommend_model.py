#!/usr/bin/env python3
"""Print a deterministic hardware-aware training recommendation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.model_sizer import estimate_total_tokens, profile_hardware, recommend_training_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend a safe Model Lab training profile.")
    parser.add_argument("--shard-dir", type=Path, default=Path("output/shards"))
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--target-hours", type=float, default=None)
    parser.add_argument("--observed-tokens-per-sec", type=float, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    hardware = profile_hardware()
    total_tokens = None
    if args.shard_dir.exists():
        try:
            total_tokens = estimate_total_tokens(args.shard_dir)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            total_tokens = None

    profile = recommend_training_profile(
        hardware,
        total_tokens=total_tokens,
        configured_steps=args.steps,
        target_training_hours=args.target_hours,
        observed_tokens_per_sec=args.observed_tokens_per_sec,
    )
    payload = {"hardware": hardware.to_dict(), "total_tokens": total_tokens, "recommendation": profile.to_dict()}
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Platform: {hardware.platform}")
    print(f"CPU cores: {hardware.cpu_cores}")
    print(f"Available RAM: {hardware.available_ram_gb:.2f} GB" if hardware.available_ram_gb is not None else "Available RAM: unknown")
    print(f"GPU: {hardware.gpu_name or 'none'} ({hardware.gpu_memory_gb:.2f} GB VRAM)")
    print(f"Hardware tier: {hardware.tier}")
    print(f"Dataset tokens: {total_tokens:,}" if total_tokens is not None else "Dataset tokens: unavailable")
    print(f"Recommendation: {profile.model_preset}, seq_len={profile.seq_len}, batch={profile.batch_size}, grad_accum={profile.grad_accum_steps}, precision={profile.precision}")
    print(f"Recommended steps: {profile.recommended_steps:,}" if profile.recommended_steps is not None else "Recommended steps: configured")
    print(f"Reason: {profile.reason}")
    if profile.estimated_hours is not None:
        print(f"Estimated duration: {profile.estimated_hours:.2f} hours")
    else:
        print("Estimated duration: unavailable until an observed throughput is supplied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
