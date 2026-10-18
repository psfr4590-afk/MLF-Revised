# Model Lab: start here

This guide is the operational walkthrough for a new installation. ML-Framework is the local-first training pipeline behind Model Lab, the **M²S Model Training Pipeline**. It intentionally tests the human path, not just the developer happy path.

## 1. Get the project

For a first-time Windows user, use GitHub's **Code → Download ZIP**, extract the archive, and open PowerShell in the extracted `ML-Framework` folder.

Run:

```powershell
.\bootstrap_windows.ps1
```

This wrapper handles the pre-Python boundary. It locates supported Python, can install Python 3.13 through `winget` when available, then hands control to the canonical `bootstrap.py`.

Git users can instead clone:

```bash
git clone https://github.com/psfr4590-afk/ML-Framework.git
cd ML-Framework
```

## 2. Install

For the ZIP path, `bootstrap_windows.ps1` already performs installation and doctor checks.

For a Git checkout or direct Python workflow:

Windows PowerShell:

```powershell
python .\bootstrap.py --install
```

Linux / macOS / Termux:

```bash
python3 bootstrap.py --install
```

The bootstrapper installs the declared dependencies and selects the CPU or CUDA PyTorch wheel automatically from detected NVIDIA hardware.

## 3. Preflight

```text
Windows: python .\bootstrap.py --doctor
Other:   python3 bootstrap.py --doctor
```

Resolve any required `FAIL` result before continuing. Missing Git is not a blocker for a ZIP download. Missing CMake is not a blocker for the basic Python smoke path, but it is required for native export.

## Canonical pipeline entrypoint

The first-boot wrapper is the human-friendly path. The underlying direct pipeline contract remains:

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

The canonical starter profile is `config/pipeline_config.yaml`.

## 4. Prove the pipeline

```text
Windows: python .\mlframework.py smoke
Other:   python3 mlframework.py smoke
```

This runs the bounded starter configuration through the real pipeline. It is a functional check, not a production training run.

Check the result:

```text
mlframework status
mlframework runs
```

If you did not install the editable package, use the equivalent Python entry points:

```text
python run_pipeline.py --doctor
python run_pipeline.py --list-stages
python run_pipeline.py --list-groups
```

## 5. Launch the Command Center

The localhost browser Command Center is the sole operator interface. `launch.py` and `ui/app.py` are compatibility launchers that delegate to the same backend.

```powershell
python .\launch.py
```

Or start the backend without opening a browser:

```powershell
python .\run_command_center.py --no-browser
```

The Command Center binds to loopback by default and reads operational state from the authoritative SQLite store.

## 6. Continue into a real run

Inspect hardware first:

```text
Windows: python .\run_pipeline.py --doctor --hardware-report
Other:   python3 run_pipeline.py --doctor --hardware-report
```

Then choose an appropriate configuration under `config/`. The starter profile is intentionally bounded and should not be mistaken for a useful production model.

The main CLI workflow is:

```text
doctor → smoke → dataset → train → evaluate → export → infer
```

## 7. Native export

Native export requires the pinned llama.cpp toolchain and CMake. On Windows, a first-boot user can prepare both with:

```powershell
.\bootstrap_windows.ps1 -Native
```

Or from an already-prepared environment:

```bash
python scripts/verify_release.py --bootstrap-native
```

Or, for the normal CLI path after a trained checkpoint exists:

```text
mlframework export
mlframework infer
```

## 8. Engineering verification

Static verification:

```bash
python scripts/verify_release.py
```

Full RC verification:

```bash
python scripts/verify_release.py --bootstrap-native --profile rc --ui-probe --clean-clone
```

See the [architecture guide](../architecture/ARCHITECTURE.md) for system design and the [release checklist](../release/RELEASE_CHECKLIST.md) for release-specific requirements.

## 9. Recovery

The bootstrap path is designed to be rerun. If dependency installation is interrupted, the network fails, or a prerequisite is missing, fix the reported problem and rerun the same command. Do not treat a failed first attempt as a reason to delete the checkout or generated provenance.

If Windows has no `winget`, install Python 3.11-3.14 manually and reopen PowerShell before rerunning `bootstrap_windows.ps1`. For native export, install CMake when the wrapper reports that it is missing.
