# System Readiness (Step 1)

System Readiness is the first operator workflow. It checks the current installation and host, identifies hard blockers and warnings, recommends a conservative training profile, and writes a machine-readable evidence report. It does not start a production pipeline.

## Run the checks

From the repository root and the project's virtual environment:

```powershell
python .\scripts\system_readiness.py
```

The command prints the report and writes `runtime/system_readiness.json`. The runtime folder is generated state and should not be committed.

Exit code `0` means required checks passed. Exit code `2` means a required check failed or the command could not complete. Optional capabilities may be warnings rather than blockers. `UNKNOWN` is not a pass.

To run a short capability benchmark against an existing, manifest-backed training shard set:

```powershell
python .\scripts\system_readiness.py --benchmark
```

The benchmark runs in a child process and is terminated if it exceeds the default 90-second wall-clock limit. This benchmark requires an existing `shards.manifest.json`; it does not crawl, clean, tokenize, or create a dataset. Its model preset and step count come from the current pipeline configuration. A benchmark failure is recorded separately from installation readiness because a failed model probe does not necessarily mean the application itself cannot run.

Use `--output PATH` to choose a report destination and `--benchmark-timeout SECONDS` to adjust the benchmark wall-clock limit.

## Status meanings

- `READY`: required checks passed and no warnings were recorded.
- `READY_WITH_LIMITS`: required checks passed, but one or more limitations or unknown measurements need attention.
- `BLOCKED`: at least one required check failed.
- Individual checks use `PASS`, `WARN`, `BLOCKED`, or `UNKNOWN`. A non-required benchmark failure is reported as a benchmark blocker but does not by itself change required installation readiness.

The report records runtime versions, detected hardware, current available RAM, disk headroom, CUDA runtime state, check details, recommended profile, optional benchmark results, and Git identity when available. It intentionally omits user names and host names. A report SHA-256 is included for integrity checking; it is not a digital signature.

## Limits

Disk-space thresholds are conservative guardrails, not predictions of the full corpus size. Hardware detection does not prove training viability. The optional benchmark is the empirical check and only covers its specific model, context, shard geometry, and step count. It does not establish model quality or production-training convergence.
