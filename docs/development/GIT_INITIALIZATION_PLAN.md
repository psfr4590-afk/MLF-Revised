# Git Initialization and Source-Control Policy

This file is the maintained source-control policy for Model Lab. It supersedes the original pre-Git initialization notes that described a repository before the public project existed.

## Track

Track source code, tests, configuration templates, release tooling, CI, documentation, and compatibility shims.

Primary tracked surfaces:

- `pipeline/`
- `command_center/`
- `ui/`
- `scripts/`
- `tests/`
- `config/`
- root entrypoints and project metadata
- `.github/`
- release and architecture documentation

## Do not track

Runtime state and machine-specific artifacts stay outside Git:

- `datasets/`
- `output/`
- `scratch/`
- `.runtime/`
- logs and caches
- checkpoints and model files
- live `.env` files and credential stores
- native llama.cpp build products
- editor and operating-system noise

The public repository should contain everything required to understand and reproduce the system, but not the potentially enormous or sensitive products created by running it.

## Configuration policy

`config/` is the canonical configuration location. Example credential and environment files are safe templates. Live credentials must be supplied through the runtime credential store or environment variables.

## Compatibility policy

Compatibility modules are allowed when they preserve an established public import path, but they must delegate to one canonical implementation. Parallel copies of core data contracts or stage implementations are prohibited because they eventually drift and then everyone gets to discover the disagreement at 2:17 AM.

## Release hygiene

Before release:

1. Run compilation and the complete test suite.
2. Run the non-destructive release verifier.
3. Run `run_pipeline.py --doctor` on the target machine.
4. Verify optional native/CUDA capabilities separately.
5. Confirm no credentials or generated runtime artifacts are staged.
6. Review `docs/release/BUILD_MANIFEST.json`, `docs/release/RELEASE_CHECKLIST.md`, and `docs/release/RELEASE_READINESS_PLAN.md`.
7. Verify the public `README.md` and `docs/development/START_HERE.md` describe the actual current entrypoints.

## Source-of-truth rule

GitHub `main` is the public source of truth. The archived delivery package is a reference for missing architecture and release evidence, not a reason to overwrite newer repository improvements wholesale.
