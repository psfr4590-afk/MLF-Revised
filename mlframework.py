#!/usr/bin/env python3
"""Clone-and-run command line interface for ML-Framework."""
from __future__ import annotations
import argparse, importlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "1.3.0"
MIN_PYTHON = (3, 11)
MAX_PYTHON_EXCLUSIVE = (3, 15)
DEFAULT_CONFIG = ROOT / "config" / "pipeline_config.yaml"
LLAMACPP_DIR = ROOT / "third_party" / "llama.cpp"
MIN_DISK_GIB = 8.0
RUNTIME_ESTIMATE = "~1h 15m (hardware-dependent operator estimate)"


def _run(command: list[str], *, cwd: Path = ROOT) -> int:
    print("$", " ".join(command))
    return subprocess.run(command, cwd=cwd, check=False).returncode


def _python() -> list[str]:
    return [sys.executable]


def _import_ok(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", "installed")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _ram_gib() -> float | None:
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                text=True, capture_output=True, check=False, timeout=5,
            )
            value = proc.stdout.strip()
            if value.isdigit():
                return round(int(value) / (1024**3), 2)
        except (OSError, subprocess.SubprocessError):
            return None
    if hasattr(os, "sysconf"):
        try:
            return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3), 2)
        except (OSError, ValueError):
            return None
    return None


def _gpu() -> tuple[bool, bool, str, float | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, False, "No NVIDIA GPU detected; CPU fallback is supported", None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True, False, "nvidia-smi is present but GPU query failed", None
    if proc.returncode != 0 or not proc.stdout.strip():
        return True, False, "nvidia-smi is present but no usable GPU was reported", None
    parts = [p.strip() for p in proc.stdout.splitlines()[0].split(",", 1)]
    vram = None
    if len(parts) == 2:
        try:
            vram = round(float(parts[1]) / 1024, 2)
        except ValueError:
            pass
    return True, True, parts[0], vram


def _torch_state() -> tuple[bool, bool, str]:
    ok, detail = _import_ok("torch")
    if not ok:
        return False, False, detail
    import torch
    cuda = bool(torch.cuda.is_available())
    if not cuda:
        return True, False, f"PyTorch {torch.__version__}; CPU fallback"
    return True, True, f"PyTorch {torch.__version__}; CUDA {torch.version.cuda or 'unknown'}"


def _disk_gib() -> float:
    return round(shutil.disk_usage(ROOT).free / (1024**3), 2)


def _git_ok() -> tuple[bool, str]:
    exe = shutil.which("git")
    if not exe:
        return False, "git executable not found on PATH"
    proc = subprocess.run([exe, "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.returncode == 0, proc.stdout.strip() if proc.returncode == 0 else "not a Git working tree"


def _llamacpp_ok() -> tuple[bool, str]:
    if not LLAMACPP_DIR.is_dir():
        return False, "pinned llama.cpp checkout not present; export will bootstrap it"
    converter = LLAMACPP_DIR / "convert_hf_to_gguf.py"
    if not converter.is_file():
        return False, "convert_hf_to_gguf.py missing"
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=LLAMACPP_DIR, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return False, "llama.cpp checkout is not a Git repository"
    quantizer = next((p for p in LLAMACPP_DIR.rglob("llama-quantize*") if p.is_file() and os.access(p, os.X_OK)), None)
    cli = None
    for name in ("llama", "llama-cli", "llama.exe", "llama-cli.exe"):
        cli = next((p for p in LLAMACPP_DIR.rglob(name) if p.is_file()), None)
        if cli:
            break
    if quantizer is None:
        return False, f"llama.cpp {proc.stdout.strip()[:12]}... present; llama-quantize not built"
    if cli is None:
        return False, "llama.cpp quantizer is built but inference CLI is not built"
    return True, f"{proc.stdout.strip()[:12]}... converter, quantizer, inference CLI ready"


def _configuration_ok() -> tuple[bool, str]:
    if not DEFAULT_CONFIG.is_file():
        return False, "config/pipeline_config.yaml is missing"
    try:
        import yaml
        from pipeline.config_validation import validate_config
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
        validate_config(cfg)
        return True, "starter configuration passed schema validation"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _dataset_ready() -> tuple[bool, str]:
    seeds = ROOT / "config" / "seed_urls.txt"
    groups = ROOT / "config" / "dataset_groups.smoke.yaml"
    if not seeds.is_file():
        return False, "config/seed_urls.txt is missing"
    if not groups.is_file():
        return False, "config/dataset_groups.smoke.yaml is missing"
    return True, "starter dataset definitions and seed URLs are present"


def _tokenizer_ready() -> tuple[bool, str]:
    ok, detail = _import_ok("tokenizers")
    if not ok:
        return False, detail
    try:
        import yaml
        cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
        vocab = int(cfg["tokenizer"]["vocab_size"])
        return True, f"tokenizers installed; starter tokenizer contract requests vocab_size={vocab}"
    except Exception as exc:
        return False, f"tokenizer configuration invalid: {exc}"


def doctor() -> int:
    checks: list[tuple[str, str, str]] = []
    py_ok = MIN_PYTHON <= sys.version_info[:2] < MAX_PYTHON_EXCLUSIVE
    checks.append(("Python", "PASS" if py_ok else "FAIL", f"{sys.version.split()[0]} (3.11-3.14 supported)"))
    torch_ok, cuda_ok, torch_detail = _torch_state()
    checks.append(("PyTorch", "PASS" if torch_ok else "FAIL", torch_detail))
    gpu_detected, gpu_ok, gpu_detail, vram = _gpu()
    checks.append(("CUDA", "PASS" if cuda_ok else ("FALLBACK" if torch_ok else "FAIL"), "torch.cuda.is_available()" if cuda_ok else "CPU fallback; CUDA is optional"))
    checks.append(("GPU", "PASS" if gpu_ok else ("FALLBACK" if not gpu_detected else "FAIL"), gpu_detail))
    checks.append(("VRAM", "PASS" if vram is not None else ("FALLBACK" if torch_ok and not gpu_detected else "FAIL"), f"{vram:.2f} GiB detected" if vram is not None else "not applicable on CPU fallback"))
    ram = _ram_gib()
    checks.append(("RAM", "PASS" if ram is not None else "FAIL", f"{ram:.2f} GiB detected" if ram is not None else "unable to measure RAM"))
    free = _disk_gib()
    checks.append(("Disk", "PASS" if free >= MIN_DISK_GIB else "FAIL", f"{free:.2f} GiB free; {MIN_DISK_GIB:.0f} GiB minimum for onboarding"))
    tok_ok, tok_detail = _tokenizer_ready()
    checks.append(("Tokenizer", "PASS" if tok_ok else "FAIL", tok_detail))
    data_ok, data_detail = _dataset_ready()
    checks.append(("Dataset", "PASS" if data_ok else "FAIL", data_detail))
    llama_ok, llama_detail = _llamacpp_ok()
    checks.append(("llama.cpp", "PASS" if llama_ok else "WARN", llama_detail))
    git_ok, git_detail = _git_ok()
    checks.append(("Git", "PASS" if git_ok else "FAIL", git_detail))
    cfg_ok, cfg_detail = _configuration_ok()
    checks.append(("Configuration", "PASS" if cfg_ok else "FAIL", cfg_detail))
    print("ML-FRAMEWORK DOCTOR")
    for name, state, detail in checks:
        print(f"{name:<14} {state:<8} {detail}")
    print()
    print("Recommended:")
    print(" This CLI no longer recommends fixed model presets.")
    print(" Training recommendations should come from gradient-testing results")
    print(" that establish the actual hardware ceiling and choke point.")
    print("Estimated runtime:")
    print(f" {RUNTIME_ESTIMATE}")
    print()
    required_failures = [name for name, state, _ in checks if state == "FAIL"]
    if required_failures:
        print(f"STATUS: NOT READY ({', '.join(required_failures)})")
        return 2
    if not llama_ok:
        print("STATUS: READY (training/smoke path ready; native export toolchain not yet built)")
    else:
        print("STATUS: READY")
    return 0


def _pipeline(*args: str) -> int:
    return _run(_python() + [str(ROOT / "run_pipeline.py"), *args])


def command_smoke() -> int:
    return _pipeline("--config", str(DEFAULT_CONFIG), "--no-resume")


def command_dataset() -> int:
    return _pipeline("--config", str(DEFAULT_CONFIG), "--stages", "crawl,clean,dedup,weight,tokenize,shard", "--no-resume")


def command_train() -> int:
    return _pipeline("--config", str(DEFAULT_CONFIG), "--stages", "train")


def command_evaluate() -> int:
    return _run(_python() + [str(ROOT / "scripts" / "evaluate_model.py"), "--checkpoint", "best"])


def command_export() -> int:
    if not LLAMACPP_DIR.is_dir():
        rc = _run(_python() + [str(ROOT / "scripts" / "reconcile_environment.py"), "--project-root", str(ROOT), "--ensure-llamacpp"])
        if rc:
            return rc
    return _run(_python() + [str(ROOT / "scripts" / "export_gguf.py"), "--output-dir", str(ROOT / "output"), "--llamacpp-dir", str(LLAMACPP_DIR), "--quant", "ALL"])


def _default_gguf() -> Path:
    manifest = ROOT / "output" / "gguf" / "export_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        artifacts = data.get("artifacts") or {}
        if "Q4_K_M" in artifacts:
            return Path(artifacts["Q4_K_M"]["artifact"]["path"])
        final = data.get("final_gguf")
        if final:
            return Path(final)
    return ROOT / "output" / "gguf" / "model-q4_k_m.gguf"


def command_infer(prompt: str) -> int:
    model = _default_gguf()
    if not model.is_file():
        print(f"Inference model not found: {model}. Run mlframework export first.")
        return 2
    command = [str(ROOT / "scripts" / "verify_gguf.py"), "--model", str(model), "--llamacpp-dir", str(LLAMACPP_DIR), "--prompt", prompt]
    # Enforce the same context-length and special-token contract as the release gate.
    manifest = ROOT / "output" / "gguf" / "export_manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        seq_len = (data.get("model_config") or {}).get("seq_len")
        ids = (data.get("export_config") or {}).get("special_token_ids") or {}
        if seq_len:
            command += ["--context-length", str(int(seq_len))]
        for role, flag in (("bos", "--bos-token-id"), ("eos", "--eos-token-id"), ("pad", "--pad-token-id")):
            if role in ids:
                command += [flag, str(int(ids[role]))]
    except (OSError, ValueError, TypeError):
        print("Warning: export manifest unreadable; running inference without context/special-token checks.")
    return _run(_python() + command)


def command_status() -> int:
    manifest_root = ROOT / "output" / "runs"
    manifests = sorted(manifest_root.glob("*/run_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True) if manifest_root.is_dir() else []
    print("ML-FRAMEWORK STATUS")
    if not manifests:
        print("No pipeline runs recorded.")
        print("Next: mlframework smoke")
        return 0
    data = json.loads(manifests[0].read_text(encoding="utf-8"))
    print(f"Run: {data.get('run_id', 'unknown')}")
    print(f"Experiment: {data.get('experiment_id', 'unknown')}")
    print(f"Dataset: {data.get('dataset_id') or 'not assigned'}")
    print(f"Status: {data.get('final_status', 'UNKNOWN')}")
    print(f"Git: {(data.get('git') or {}).get('commit', 'unknown')}")
    print("Stages:")
    for name, stage in (data.get("stages") or {}).items():
        if isinstance(stage, dict):
            print(f"  {name:<12} {stage.get('status', 'UNKNOWN')}")
    return 0


def command_runs() -> int:
    db_path = ROOT / "output" / "experiment.db"
    print("ML-FRAMEWORK RUNS")
    if not db_path.is_file():
        print("No experiment database yet.")
        return 0
    from pipeline.experiment_db import ExperimentDB
    db = ExperimentDB(db_path)
    try:
        rows = db.conn.execute("SELECT id,experiment_id,dataset_id,started_at,completed_at,status,git_sha FROM runs ORDER BY started_at DESC LIMIT 20").fetchall()
        if not rows:
            print("No runs recorded.")
            return 0
        for row in rows:
            print(f"{row['id']}  {row['status']:<10} {row['started_at']}  dataset={row['dataset_id'] or '-'}  git={row['git_sha'] or '-'}")
    finally:
        db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlframework", description="ML-Framework clone-and-run CLI")
    parser.add_argument("--version", action="version", version=f"ML-Framework {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check the host and project without modifying runtime artifacts")
    sub.add_parser("smoke", help="run the bounded starter pipeline")
    sub.add_parser("dataset", help="build the starter dataset through sharding")
    sub.add_parser("train", help="train using the canonical starter configuration")
    sub.add_parser("evaluate", help="evaluate the best available checkpoint")
    sub.add_parser("export", help="export F16/Q4_K_M/Q5_K_M/Q8_0 using pinned llama.cpp")
    infer = sub.add_parser("infer", help="run llama.cpp inference against the exported Q4_K_M model")
    infer.add_argument("--prompt", default="Hello")
    sub.add_parser("status", help="show the latest run manifest")
    sub.add_parser("runs", help="list recent SQLite experiment runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"doctor": doctor, "smoke": command_smoke, "dataset": command_dataset, "train": command_train, "evaluate": command_evaluate, "export": command_export, "infer": lambda: command_infer(args.prompt), "status": command_status, "runs": command_runs}
    return int(handlers[args.command]())


if __name__ == "__main__":
    raise SystemExit(main())
