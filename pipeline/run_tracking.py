"""Authoritative run identity and stage observability for Model Lab."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.experiment_db import ExperimentDB


STATES = {"PENDING", "RUNNING", "PASS", "WARN", "FAILED", "SKIPPED", "DEGRADED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_path(path: Path) -> str | None:
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(bytes.fromhex(_hash_path(child) or "0" * 64))
        return digest.hexdigest()
    return None


def _path_record(path: Path, root: Path) -> dict[str, Any]:
    try:
        display = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        display = path.resolve().as_posix()
    return {"path": display, "sha256": _hash_path(path)}


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or "").strip() if result.returncode == 0 else None


def _git_state(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status_entries": status.splitlines() if status else [],
    }


def _environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
    }


def _hardware() -> dict[str, Any]:
    try:
        import bootstrap
        return bootstrap.detect_hardware().as_dict()
    except Exception as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
        available = bool(torch.cuda.is_available())
        return {
            "version": getattr(torch, "__version__", None),
            "cuda_available": available,
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "device_count": int(torch.cuda.device_count()) if available else 0,
            "devices": [
                {"index": i, "name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))}
                for i in range(torch.cuda.device_count())
            ] if available else [],
        }
    except Exception as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


class RunTracker:
    """Own and durably update one authoritative run manifest."""

    schema = 1

    def __init__(
        self,
        root: Path,
        output_dir: Path,
        *,
        run_id: str | None,
        dataset_id: str,
        experiment_id: str,
        configuration: dict[str, Any],
        dataset_identity: dict[str, Any] | None = None,
        source_identity: dict[str, Any] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.output_dir = output_dir.resolve()
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:12]}"
        self.dataset_id = str(dataset_id)
        self.experiment_id = str(experiment_id)
        self.run_dir = self.output_dir / "runs" / self.run_id
        self.path = self.run_dir / "run_manifest.json"
        self.database_path = self.output_dir / "experiment.db"
        self.db = ExperimentDB(self.database_path)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        now = _now()
        self.manifest = {
            "schema": self.schema,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "timestamps": {"started": now, "updated": now, "ended": None},
            "git": _git_state(self.root),
            "environment": _environment(),
            "hardware": {"host": _hardware(), "torch": _torch_environment()},
            "configuration": configuration,
            "dataset_identity": dataset_identity or {},
            "source_identity": source_identity or {},
            "source_observability": [],
            "stages": {},
            "artifacts": [],
            "metrics": {},
            "warnings": [],
            "errors": [],
            "retries": [],
            "degraded_stages": [],
            "dataset_report": None,
            "final_status": "RUNNING",
        }
        self._write()

    @classmethod
    def resume(cls, root: Path, output_dir: Path, manifest_path: Path) -> "RunTracker":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("schema") != cls.schema:
            raise RuntimeError(f"Unsupported run manifest schema: {manifest_path}")
        obj = cls.__new__(cls)
        obj.root = root.resolve()
        obj.output_dir = output_dir.resolve()
        obj.run_id = str(data["run_id"])
        obj.dataset_id = str(data["dataset_id"])
        obj.experiment_id = str(data["experiment_id"])
        obj.run_dir = manifest_path.parent
        obj.path = manifest_path
        obj.database_path = obj.output_dir / "experiment.db"
        obj.db = ExperimentDB(obj.database_path)
        obj.manifest = data
        obj.manifest["timestamps"]["updated"] = _now()
        obj.manifest["final_status"] = "RUNNING"
        obj._write()
        return obj

    def close(self) -> None:
        """Close the SQLite connection owned by this RunTracker."""
        db = getattr(self, "db", None)
        self.db = None
        if db is not None:
            db.close()

    def _write(self) -> None:
        self.manifest["timestamps"]["updated"] = _now()
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
        self.db.sync_manifest(self.manifest)

    def _record_paths(self, values: list[Path]) -> list[dict[str, Any]]:
        records = []
        for value in values:
            record = _path_record(value, self.root)
            if record["sha256"] is not None:
                self.manifest["artifacts"].append({"run_id": self.run_id, "stage": self._active_stage, **record})
            records.append(record)
        return records

    def declare_stage(self, name: str) -> None:
        self.manifest["stages"].setdefault(name, {
            "run_id": self.run_id, "status": "PENDING", "start": None, "end": None,
            "duration_seconds": None, "warnings": [], "errors": [], "inputs": [],
            "outputs": [], "artifact_hashes": {},
        })

    def start_stage(self, name: str, *, inputs: list[Path] | None = None) -> None:
        self.declare_stage(name)
        self._active_stage = name
        self.manifest["stages"][name].update({
            "status": "RUNNING", "start": _now(), "end": None, "duration_seconds": None,
            "warnings": [], "errors": [], "inputs": self._record_paths(inputs or []),
            "outputs": [], "artifact_hashes": {},
        })
        self._write()

    def finish_stage(
        self, name: str, *, status: str = "PASS", outputs: list[Path] | None = None,
        warnings: list[str] | None = None, errors: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if status not in STATES - {"PENDING", "RUNNING"}:
            raise ValueError(f"Invalid terminal stage status: {status}")
        stage = self.manifest["stages"].get(name)
        if not stage or not stage.get("start"):
            raise RuntimeError(f"Stage was never started: {name}")
        stage["status"] = status
        stage["end"] = _now()
        stage["duration_seconds"] = max(
            0.0,
            (datetime.fromisoformat(stage["end"]) - datetime.fromisoformat(stage["start"])).total_seconds(),
        )
        stage["warnings"] = list(warnings or [])
        stage["errors"] = list(errors or [])
        records = self._record_paths(outputs or [])
        stage["outputs"] = records
        stage["artifact_hashes"] = {item["path"]: item["sha256"] for item in records if item.get("sha256")}
        if metrics:
            self.manifest["metrics"].setdefault(name, {}).update(metrics)
        self.manifest["warnings"].extend(stage["warnings"])
        self.manifest["errors"].extend(stage["errors"])
        if status == "DEGRADED" and name not in self.manifest["degraded_stages"]:
            self.manifest["degraded_stages"].append(name)
        self._write()

    def fail_stage(self, name: str, exc: BaseException) -> None:
        self.finish_stage(name, status="FAILED", errors=[f"{type(exc).__name__}: {exc}"])

    def add_retry(self, stage: str, attempt: int, reason: str) -> None:
        self.manifest["retries"].append({"stage": stage, "attempt": int(attempt), "reason": str(reason), "timestamp": _now()})
        self._write()

    def finish_run(self, status: str = "PASS") -> None:
        """Finalize the run exactly once; repeated calls are harmless."""
        if status not in {"PASS", "WARN", "FAILED", "DEGRADED"}:
            raise ValueError(f"Invalid final run status: {status}")
        if self.manifest.get("timestamps", {}).get("ended") is not None:
            return
        self.manifest["final_status"] = status
        self.manifest["timestamps"]["ended"] = _now()
        self._write()
