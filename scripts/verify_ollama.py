#!/usr/bin/env python3
"""Verify a GGUF through Ollama using the generated Modelfile."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout)


def verify_ollama(
    model: str | Path,
    modelfile: str | Path,
    ollama: str = "ollama",
    model_name: str = "model-lab-rc",
    prompt: str = "Hello",
) -> dict[str, Any]:
    model_path = Path(model).resolve()
    modelfile_path = Path(modelfile).resolve()
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise FileNotFoundError(f"GGUF model is missing or empty: {model_path}")
    if not modelfile_path.is_file():
        raise FileNotFoundError(f"Modelfile is missing: {modelfile_path}")

    executable = shutil.which(ollama) or (str(Path(ollama).resolve()) if Path(ollama).is_file() else None)
    if not executable:
        raise FileNotFoundError(f"Ollama executable not found: {ollama}")

    modelfile_text = modelfile_path.read_text(encoding="utf-8")
    if f"FROM {model_path.name}" not in modelfile_text:
        raise RuntimeError(f"Modelfile does not reference the exported GGUF filename: {model_path.name}")

    with tempfile.TemporaryDirectory(prefix="model-lab-ollama-") as tmp:
        temp_model = Path(tmp) / model_path.name
        temp_modelfile = Path(tmp) / "Modelfile"
        temp_model.symlink_to(model_path)
        temp_modelfile.write_text(modelfile_text, encoding="utf-8")

        create = _run([executable, "create", model_name, "-f", str(temp_modelfile)], timeout=300)
        if create.returncode != 0:
            detail = (create.stderr or create.stdout)[-3000:]
            raise RuntimeError(f"ollama create failed with exit code {create.returncode}: {detail}")

        run = _run([executable, "run", model_name, prompt], timeout=180)
        output = (run.stdout or "").strip()
        if run.returncode != 0:
            detail = (run.stderr or run.stdout)[-3000:]
            raise RuntimeError(f"ollama run failed with exit code {run.returncode}: {detail}")
        if not output:
            raise RuntimeError("Ollama loaded the model but generated no visible output")

        show = _run([executable, "show", model_name], timeout=60)
        if show.returncode != 0:
            detail = (show.stderr or show.stdout)[-2000:]
            raise RuntimeError(f"ollama show failed with exit code {show.returncode}: {detail}")

    return {
        "status": "passed",
        "model": str(model_path),
        "modelfile": str(modelfile_path),
        "ollama": executable,
        "model_name": model_name,
        "prompt": prompt,
        "generated_output": output,
        "checks": {
            "gguf_input": "passed",
            "modelfile": "passed",
            "ollama_create": "passed",
            "ollama_load": "passed",
            "generation": "passed",
            "model_show": "passed",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify GGUF -> Modelfile -> Ollama -> generation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--modelfile", required=True)
    parser.add_argument("--ollama", default="ollama")
    parser.add_argument("--model-name", default="model-lab-rc")
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args(argv)
    try:
        result = verify_ollama(args.model, args.modelfile, args.ollama, args.model_name, args.prompt)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"OLLAMA VERIFICATION FAILED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
