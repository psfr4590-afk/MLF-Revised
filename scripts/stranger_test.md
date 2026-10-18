# Clean-clone acceptance test

Run this procedure from a directory that does not contain a previous Model Lab checkout, virtual environment, datasets, or generated artifacts.

## 1. Clone

```bash
git clone https://github.com/psfr4590-afk/ML-Framework.git
cd ML-Framework
```

## 2. Install and inspect

Windows:

```powershell
python .\bootstrap.py --install
python .\bootstrap.py --doctor
python .\run_pipeline.py --list-stages
python .\run_pipeline.py --list-groups
python .\run_pipeline.py --doctor
```

Linux/macOS/Termux:

```bash
python3 bootstrap.py --install
python3 bootstrap.py --doctor
python3 run_pipeline.py --list-stages
python3 run_pipeline.py --list-groups
python3 run_pipeline.py --doctor
```

Expected result: installation completes without repository-local prerequisites, both inspection commands are read-only, the environment doctor reports the actual host state, and the canonical starter configuration is identified as `config/pipeline_config.yaml`.

## 3. Run the starter workflow

```bash
python run_pipeline.py --no-resume
```

On systems where the executable is named `python3`, use `python3 run_pipeline.py --no-resume` instead.

Confirm that the run starts from the documented starter profile, progresses through the configured stages, writes artifacts under the documented runtime/output locations, and ends with a clear success or actionable failure message.

## 4. Repeat

Run the same command a second time. Confirm that resume behavior is deterministic and does not silently consume stale artifacts.

## 5. Package entry point

After package installation, verify:

```bash
mlab --help
```

The command should resolve to the same canonical pipeline CLI rather than a second implementation.

## 6. Release gate

```bash
python scripts/verify_release.py
```

This must report syntax, tests, and required environment checks as passing. Native llama.cpp prerequisites are intentionally separate unless `--bootstrap-native` is supplied.

## Acceptance rule

A stranger test is not complete because unit tests pass. It is complete when a fresh checkout can follow the documented path without developer-only files, undocumented manual fixes, or unexplained state carried over from another installation.
