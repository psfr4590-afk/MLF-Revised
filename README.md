# ML-Framework

**ML-Framework** is the local-first training pipeline behind **Model Lab**, the **M²S Model Training Pipeline**. It takes configured data sources through cleaning, semantic deduplication, weighting, tokenization, sharding, training, and GGUF export, with provenance and artifact integrity tracked along the way.

It is designed to run locally on Windows, Linux, macOS, and headless Termux. The operator surface is a single localhost browser Command Center; the pipeline and backend are portable.

## What you get

- End-to-end data and training pipeline
- Deterministic artifact manifests and run provenance
- Semantic near-duplicate detection
- Hardware/CUDA preflight and conservative training profiles
- Local FastAPI command center
- GGUF export and llama.cpp inference verification
- SQLite experiment/run history
- Automated tests, linting, security checks, and release verification

The normal pipeline is:

`crawl → clean → dedup → weight → tokenize → shard → train → export`

**Ten dataset groups are preconfigured** in `config/dataset_groups.yaml`, with the canonical seeded groups exposed through the Model Lab dataset surface.

## Quick start

You do not need to read the project history to use the project. Humanity has suffered enough documentation archaeology already.

### 1. Choose how you arrived here

**New to Git or using a fresh Windows laptop:** download the repository with **Code → Download ZIP**, extract it, open PowerShell in the extracted folder, and run:

```powershell
.\bootstrap_windows.ps1
```

The Windows first-boot wrapper handles the human/machine boundary before Python can run. It finds Python 3.11-3.14, can install Python 3.13 with `winget` when Python is absent, installs the project dependencies, selects CPU/CUDA PyTorch, and runs the bootstrap doctor.

**Developer path:** if Git is already installed:

```bash
git clone https://github.com/psfr4590-afk/ML-Framework.git
cd ML-Framework
```

Then run the canonical bootstrap directly:

Windows:
```powershell
python .\bootstrap.py --install
```

Linux / macOS / Termux:
```bash
python3 bootstrap.py --install
```


The bootstrapper installs the project dependencies and the PyTorch wheel appropriate to the detected host. Use `--torch-channel cpu` or `--torch-channel cuda` to override automatic selection.

### Canonical pipeline entrypoint

The CLI wrapper above is the friendlier first-boot path. The underlying pipeline entrypoint remains available and is the canonical direct-run contract:

Windows:
```powershell
python .\run_pipeline.py --doctor
python .\run_pipeline.py --no-resume
```

Linux / macOS / Termux:
```bash
python3 run_pipeline.py --doctor
python3 run_pipeline.py --no-resume
```

The starter profile is `config/pipeline_config.yaml`.

### 3. Check the machine

Windows:

```powershell
python .\bootstrap.py --doctor
```

Linux / macOS / Termux:

```bash
python3 bootstrap.py --doctor
```

The doctor checks Python, required packages, Git/CMake, hardware, PyTorch/CUDA, and the project environment. A missing llama.cpp checkout is a warning before export, not a blocker for the basic smoke run.

### 4. Run the starter pipeline

Windows:

```powershell
python .\mlframework.py smoke
```

Linux / macOS / Termux:

```bash
python3 mlframework.py smoke
```

The starter profile is deliberately small. Its purpose is to prove that the real pipeline and artifact chain work on the current machine, not to train a useful production model.

Afterward:

```text
mlframework status
mlframework runs
```

For a first-time Windows user, the shortest useful path is therefore:

```text
Download ZIP → extract → bootstrap_windows.ps1 → mlframework smoke
```

For developers:

```text
clone → bootstrap.py --install → bootstrap.py --doctor → mlframework smoke
```

See [First boot](docs/development/FIRST_BOOT.md) for the non-happy-path onboarding and recovery behavior.

## Launch Model Lab

### Command Center

There is exactly one operator interface: the localhost browser Command Center. The `launch.py` launcher and `ui/app.py` compatibility entry point both delegate to it, so they cannot drift into separate UI state.

Launch it with:

```powershell
python .\run_command_center.py
```

To start the backend without opening a browser:

```powershell
python .\run_command_center.py --no-browser
```

The command center binds to localhost by default.

## CLI

The CLI is an execution and automation interface, not a second operator UI. It writes authoritative run state to the same SQLite experiment store consumed by the Command Center.

Install the editable package if you want the `mlframework` command available directly in your shell:

```bash
python -m pip install -e .
```

Then:

```text
mlframework doctor
mlframework smoke
mlframework dataset
mlframework train
mlframework evaluate
mlframework export
mlframework infer
mlframework status
mlframework runs
```

The operator workflow is intentionally one-way: `pipeline/CLI execution → SQLite authoritative state → Command Center API → browser`.

The Python entry points remain available when you do not install the package:

```text
python bootstrap.py
python run_pipeline.py
python launch.py
python run_command_center.py
```

Use `python3` instead of `python` on systems where that is the local convention.

## Configuration

The starter configuration is:

```text
config/pipeline_config.yaml
```

It is the canonical first-run profile. Larger or specialized configurations are under `config/`.

Useful inspection commands:

```text
mlframework doctor
python run_pipeline.py --list-stages
python run_pipeline.py --list-groups
```

Runtime paths are resolved from the repository root. Do not copy machine-specific absolute paths into configuration.

## Project layout

```text
ML-Framework/
├── README.md
├── bootstrap.py             # Environment setup and hardware preflight
├── mlframework.py           # User-facing CLI
├── run_pipeline.py          # Pipeline entry point
├── launch.py                # Canonical Command Center launcher
├── run_command_center.py    # Local Command Center server
├── config/                  # Starter and specialized configurations
├── pipeline/                # Pipeline implementation
├── command_center/          # API and single browser operator surface
├── scripts/                 # Verification, export, and tooling
├── tests/                   # Automated verification
└── docs/                    # Architecture, development, and release docs
```

The Command Center calls the existing pipeline/backend services. It does not maintain a second implementation of the crawler, tokenizer, trainer, or exporter. The legacy `ui/` package is retained only as a compatibility launcher and is not a second operator interface.

## Documentation

Start with:

- [First boot](docs/development/FIRST_BOOT.md)
- [Development guide](docs/development/START_HERE.md)
- [Architecture](docs/architecture/ARCHITECTURE.md)
- [Project state](docs/development/PROJECT_STATE.md)
- [Dataset provenance](docs/development/DATA_PROVENANCE.md)

For engineering and release work:

- [Verification](docs/verification/VERIFICATION.md)
- [Verification checklist](docs/verification/VERIFICATION_CHECKLIST.md)
- [Release checklist](docs/release/RELEASE_CHECKLIST.md)
- [Release readiness](docs/release/RELEASE_READINESS_PLAN.md)
- [Release verification](docs/release/RELEASE_VERIFICATION_REPORT.md)

## Development verification

Run the normal static verification with:

```bash
python scripts/verify_release.py
```

For the full native release gate, including pinned llama.cpp bootstrap and GGUF/inference verification:

```bash
python scripts/verify_release.py --bootstrap-native
```

The full RC gate can additionally exercise the UI/backend probe and clean-clone onboarding path:

```bash
python scripts/verify_release.py --bootstrap-native --profile rc --ui-probe --clean-clone
```

These commands are verification tools, not required steps for a newcomer who only wants to run the starter pipeline.

## Requirements

- Python 3.11-3.14 for the runtime
- Git is optional for ZIP users and required only for the clone/developer workflow
- CMake is required for native export/toolchain work
- Windows ZIP first boot uses PowerShell and can use `winget` to install missing Python/CMake prerequisites
- NVIDIA CUDA is optional
- CPU mode is supported for bounded smoke/testing runs
- At least 8 GiB free disk space for onboarding; real training can require substantially more
- `sentence-transformers` and `faiss-cpu` are installed because semantic deduplication is part of the pipeline

For serious training, run the doctor/hardware report before choosing a larger configuration.

## Release boundary

A passing source tree does not mean that a production dataset has been crawled, a useful model has converged, external-source training rights have been certified, or target hardware has passed native inference. Those claims require runtime evidence.

The repository keeps those release checks in the verification tooling and release documentation rather than presenting historical results as current facts.

## License

See [LICENSE](LICENSE).

