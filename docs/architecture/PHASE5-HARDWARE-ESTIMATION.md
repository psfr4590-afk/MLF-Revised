# Phase 5: Hardware Detection and Training Estimation

## Goal

A training run should be evaluated for viability before it commits substantial compute.

Phase 5 makes the decision observable:

```
Requested configuration
        ↓
Hardware snapshot
        ↓
Conservative auto-sizing
        ↓
Preflight benchmark
        ↓
Effective configuration
        ↓
Estimated duration / checkpoints / evaluations
        ↓
Training with live ETA
```

## Hardware snapshot

Each training preflight records:

- CPU name, cores, and threads
- total and available RAM
- GPU name and VRAM
- CUDA availability and runtime version
- NVIDIA driver when available
- PyTorch version
- operating system
- total and free disk space

Hardware facts are detected at runtime. GPU presence alone is not treated as proof that PyTorch CUDA is usable.

## Auto-sizing

When `train.auto_size=true`, the framework records both the requested training configuration and the effective configuration selected for the host.

The low-VRAM contract is intentionally conservative. For a detected GPU below 4 GiB, the default profile is:

- model: 85M
- sequence length: 128
- batch size: 1
- gradient accumulation: 32
- precision: FP16

The sizing decision records machine-readable reasons such as `VRAM < 4 GB` and shard geometry caps.

## Preflight benchmark

Before long training, a fresh model and optimizer execute a short real training benchmark against the existing training shards.

Measured values include:

- tokens/sec
- steps/sec
- peak PyTorch GPU memory
- GPU utilization when NVIDIA telemetry is available
- benchmark duration

A failed benchmark, including CUDA out-of-memory, rejects the training run before the long training loop begins.

The report is written to `output/preflight_report.json` and contains requested configuration, effective configuration, hardware, benchmark results, and estimates.

## Duration and live ETA

The preflight throughput is used to estimate:

- total duration
- evaluation schedule
- checkpoint schedule
- expected completion time

During training, `output/logs/metrics.jsonl` records:

- elapsed time
- actual tokens/sec
- estimated remaining time
- live completion estimate
- initial preflight estimate
- actual time elapsed

Checkpoint metadata also records the initial estimate and estimated-vs-actual duration.

## Exit gate

The Phase 5 exit gate is satisfied when a training run can answer:

1. What hardware was detected?
2. What configuration was requested?
3. What configuration was actually selected?
4. Why was it selected?
5. Did the selected configuration survive a real preflight benchmark?
6. What throughput was measured?
7. How long should the run take?
8. When should checkpoints and evaluations occur?
9. How does the live estimate compare with the original estimate?

This is an estimate, not prophecy. Hardware contention, thermal throttling, filesystem performance, kernel selection, and other runtime effects can move the actual duration.
