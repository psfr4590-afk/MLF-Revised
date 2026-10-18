# Model Lab — Project State

**Version:** 1.3.0  
**Release posture:** RC source-ready; production release unverified  
**Source of truth:** the current public `main` branch

## Identity

- Package: **Model Lab**
- System: **M²S Model Training Pipeline**
- `bootstrap.py`: environment installation and environment preflight (`--install`, `--doctor`)
- `run_pipeline.py` / `mlab`: canonical pipeline CLI and project/runtime readiness doctor
- `launch.py`: canonical launcher for the single localhost browser Command Center
- `run_command_center.py`: localhost FastAPI Command Center launcher
- Pipeline: crawl → clean → dedup → weight → tokenize → shard → train → export

## Dataset catalog

The canonical checked-in dataset catalog contains ten groups:

1. `swe_cs_systems` — Software Engineering + CS + Systems
2. `ai_ml_cybersec_dataeng` — AI/ML + Cybersecurity + Data Engineering
3. `sci_reasoning_forensics_formal` — Scientific Reasoning + Forensics + Formal Methods
4. `domain_finance_bio_robotics` — Finance + Biology + Robotics
5. `math_statistics_optimization` — Mathematics + Statistics + Optimization
6. `physics_chemistry_materials` — Physics + Chemistry + Materials Science
7. `biomedical_health_science` — Biomedical Science + Bioinformatics + Health Research
8. `law_compliance_governance` — Law + Compliance + Digital Governance
9. `linguistics_information_retrieval` — Linguistics + NLP + Information Retrieval
10. `climate_energy_geospatial` — Climate Science + Energy + Geospatial Analysis

General web seeds live in `config/seed_urls.txt`.

## Architecture

The browser Command Center is the sole operator interface. `launch.py` and `ui/app.py` are compatibility launchers that delegate to `run_command_center.py`; dataset state, stage execution, credentials, and runtime behavior are owned by the existing backend.

The public repository intentionally does not vendor generated datasets, checkpoints, GGUF artifacts, caches, live credentials, or native build products.

## Diagnostics

`bootstrap.py --doctor` checks installation and host prerequisites without running pipeline stages. `run_pipeline.py --doctor` checks project/runtime readiness. `--hardware-report` adds conservative host-specific training guidance. These are complementary checks.

## Release controls

The current release controls include:

- Linux and Windows CI;
- PowerShell bootstrap contract validation;
- security scanning and dependency auditing;
- deterministic artifact/provenance contracts;
- pinned llama.cpp bootstrap;
- native GGUF export and inference verification;
- release dependency freeze and SBOM generation;
- dataset and model-card generation;
- release evidence packaging.

## Current release boundary

The repository is maintained as an RC source candidate, not represented as a production release. Current CI/security/CodeQL results must always be read from GitHub for the exact commit under review.

The implementation and repository-side release machinery are present. The remaining release evidence is execution-dependent:

1. CI, security, and CodeQL must be green for the exact RC commit.
2. `python scripts/verify_release.py --bootstrap-native` must pass.
3. Native GGUF inference validation must be recorded as passed.
4. Release evidence must include dependency freeze, SBOM, provenance, export manifest, integrity checks, and inference results.
5. A real production training run remains a separate operational validation and is not implied by the bounded release smoke.

Historical verification reports remain useful as audit history but do not override current code or current release evidence. The release checklist is a procedure, not a permanent record of current PASS state.

## Native dependency

llama.cpp is not vendored in the public repository. The supported native revision is bootstrapped separately when GGUF conversion, quantization, or native inference verification is required. On 2026-09-29, the pinned revision was independently built and executed on Android 16/aarch64 through the `llama-quantize` target. The resulting executable was located at `third_party/llama.cpp/build-model-lab/bin/llama-quantize` and successfully started its help output. This is recorded as native-toolchain evidence only; the complete Python release gate remains separate.

## Repository hygiene

Generated runtime outputs belong outside the source contract:

- datasets
- output artifacts
- scratch data
- checkpoints
- logs
- caches
- native build products
- live credentials and `.env` files

The repository should remain reproducible from a clean checkout plus its documented dependencies and release procedures.
