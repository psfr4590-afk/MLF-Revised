from __future__ import annotations

import os
import subprocess
import sys
import threading
import json
import base64
import urllib.parse
from pathlib import Path

try:
    import psutil
except ImportError:  # optional; without it, recovery fails closed for live PIDs
    psutil = None

from .config import ROOT
from .secrets import credentials
from .store import store

RUNS: dict[int, subprocess.Popen | object] = {}
CANCEL_REQUESTED: set[int] = set()
STOP_TIMEOUT_SECONDS = 5
LOCK = threading.RLock()
_STARTING = object()
VALID_STAGES = {"crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export"}


def _state_path(did: int) -> Path:
    return store.path(did) / "runner_state.json"


def _process_identity(pid: int | None) -> dict | None:
    if not pid or psutil is None:
        return None
    try:
        proc = psutil.Process(pid)
        return {"pid": pid, "create_time": proc.create_time()}
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def _write_state(did: int, *, stage: str, pid: int | None, status: str) -> None:
    path = _state_path(did)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema": 2, "dataset_id": did, "stage": stage, "pid": pid, "process_identity": _process_identity(pid), "status": status}, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _clear_state(did: int) -> None:
    try:
        _state_path(did).unlink()
    except FileNotFoundError:
        pass


def _sensitive_environment_values(env: dict[str, str]) -> list[str]:
    """Return values from environment variables that conventionally hold secrets."""
    import re
    sensitive_name = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|PRIVATE[_-]?KEY|CREDENTIAL)", re.I)
    return [value for name, value in env.items() if sensitive_name.search(name) and isinstance(value, str) and value]


def _redact_text(value: str, secrets: list[str] | tuple[str, ...]) -> str:
    """Redact raw secrets and common textual encodings before logging."""
    candidates: set[str] = set()
    for secret in secrets:
        if not isinstance(secret, str) or not secret:
            continue
        candidates.add(secret)
        candidates.add(json.dumps(secret, ensure_ascii=False))
        candidates.add(base64.b64encode(secret.encode()).decode())
        candidates.add(base64.urlsafe_b64encode(secret.encode()).decode())
        candidates.add(urllib.parse.quote(secret, safe=""))
        candidates.add(urllib.parse.quote_plus(secret))
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate:
            value = value.replace(candidate, "[REDACTED]")
    return value


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _same_process_identity(state: dict) -> bool:
    pid = state.get("pid")
    identity = state.get("process_identity")
    if not _pid_alive(pid):
        return False
    # Preserve a live PID when identity evidence is unavailable. This avoids
    # killing/reclassifying an unrelated process; emit a diagnostic in future.
    # PID-reuse detection is only definitive when creation-time evidence exists.
    if not isinstance(identity, dict) or identity.get("pid") != pid or psutil is None:
        return True
    try:
        return abs(psutil.Process(pid).create_time() - float(identity["create_time"])) < 0.01
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError, ValueError):
        return False


def reconcile_orphaned_runs() -> None:
    """Recover persisted RUNNING state after a Command Center restart."""
    for dataset in store.list():
        if not dataset or dataset.get("status") != "RUNNING":
            continue
        state_path = _state_path(int(dataset["id"]))
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            state = {}
        if _same_process_identity(state):
            continue
        did = int(dataset["id"])
        stage = str(state.get("stage") or next((name for name, value in dataset.get("stages", {}).items() if value == "running"), "unknown"))
        stages = dict(dataset.get("stages") or {})
        if stage in VALID_STAGES and stages.get(stage) == "running":
            stages[stage] = "failed"
        store.update(did, stages=stages, status="ERROR")
        store.event(did, "stage.recovered_orphan", {"stage": stage, "reason": "command-center restarted and worker PID is no longer alive"})
        _clear_state(did)


def _run(did, stage):
    """Run a stage and recover cleanly from otherwise-unhandled worker errors."""
    try:
        _run_inner(did, stage)
    except Exception as exc:
        # Cleanup must still happen if setup fails before _run_inner reaches
        # its subprocess try/finally. Never log exception text here.
        with LOCK:
            RUNS.pop(did, None)
            CANCEL_REQUESTED.discard(did)
            try:
                _clear_state(did)
            except Exception:
                pass

        # State persistence is best-effort: the store itself may be failing.
        # Never overwrite a terminal stage result if a later refresh/cleanup
        # operation raised after the stage had already finished.
        try:
            current = store.get(did)
            if current:
                stages = dict(current.get("stages") or {})
                existing_stage = stages.get(stage)
                already_terminal = existing_stage in {"complete", "cancelled", "failed"}

                if not already_terminal and stage in VALID_STAGES:
                    stages[stage] = "failed"

                if not already_terminal:
                    status = (
                        "CANCELLED"
                        if existing_stage == "cancelled"
                        else "ERROR"
                    )
                    store.update(did, stages=stages, status=status)
                elif stages != dict(current.get("stages") or {}):
                    store.update(did, stages=stages)
                store.event(
                    did,
                    "stage.worker_error" if already_terminal else "stage.setup_failed",
                    {"stage": stage, "error_type": type(exc).__name__},
                )
        except Exception:
            # Preserve the original worker failure and never leak its message.
            pass


def _run_inner(did, stage):
    d = store.get(did)
    stages = dict(d["stages"])
    stages[stage] = "running"
    store.update(did, stages=stages, status="RUNNING")
    store.event(did, "stage.started", {"stage": stage})
    env = os.environ.copy()
    env.update({"DATASET_ID": str(did), "DATASET_DIR": str(store.path(did)), "PROJECT_ROOT": str(ROOT)})
    redacted_values: list[str] = []
    try:
        credential_env = credentials.environment()
        env.update(credential_env)
    except Exception:
        # Do not echo credential-store exceptions; messages can expose sensitive context.
        credential_env = {}
    # Child processes inherit the caller environment, not just stored credentials.
    redacted_values = _sensitive_environment_values(env)
    def _redact_log_line(value: str) -> str:
        return _redact_text(value, redacted_values)
    cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--dataset-id", str(did), "--stages", stage]
    logp = store.path(did) / "logs" / "command-center.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK:
            cancelled_before_launch = did in CANCEL_REQUESTED
        if cancelled_before_launch:
            current = store.get(did)
            stages = dict(current["stages"])
            stages[stage] = "cancelled"
            store.update(did, stages=stages, status="CANCELLED")
            store.event(did, "stage.cancelled", {"stage": stage, "phase": "before_launch"})
            return
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", start_new_session=(os.name == "posix"), creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
        with LOCK:
            RUNS[did] = p
            _write_state(did, stage=stage, pid=getattr(p, "pid", None), status="running")
            cancel_after_launch = did in CANCEL_REQUESTED
        if cancel_after_launch:
            _terminate_process_tree(p)
        for line in p.stdout or []:
            with logp.open("a", encoding="utf-8") as f:
                f.write(_redact_log_line(line.rstrip()) + "\n")
        code = p.wait()
        current = store.get(did)
        stages = dict(current["stages"])
        with LOCK:
            was_cancelled = did in CANCEL_REQUESTED
        stages[stage] = "cancelled" if was_cancelled else ("complete" if code == 0 else "failed")
        if was_cancelled:
            store.update(did, stages=stages, status="CANCELLED")
            store.event(did, "stage.cancelled", {"stage": stage, "exit_code": code})
        elif code == 0:
            store.update(did, stages=stages, status=current.get("status", "RUNNING"))
            store.refresh_pipeline_state(did)
            refreshed = store.get(did) or current
            store.event(did, "stage.completed", {
                "stage": stage,
                "status": refreshed.get("status"),
                "stats": refreshed.get("stats", {}),
            })
        else:
            store.update(did, stages=stages, status="ERROR")
            store.refresh_stats(did)
            store.event(did, "stage.failed", {"stage": stage, "exit_code": code})
    except OSError as exc:
        current = store.get(did)
        stages = dict(current["stages"])
        stages[stage] = "failed"
        store.update(did, stages=stages, status="ERROR")
        store.refresh_stats(did)
        # Exception text may expose paths, command arguments, or credentials.
        store.event(did, "stage.launch_failed", {"stage": stage, "error_type": type(exc).__name__})
    finally:
        with LOCK:
            RUNS.pop(did, None)
            CANCEL_REQUESTED.discard(did)
            _clear_state(did)
        # Reconcile artifacts after the process exits, but preserve an explicit
        # failed stage rather than allowing refresh to hide it behind STALE/IN_PROGRESS.
        final = store.get(did)
        if final and final.get("status") != "CANCELLED" and final.get("stages", {}).get(stage) not in {"failed", "cancelled"}:
            store.refresh_pipeline_state(did)


def start_stage(did, stage):
    if stage not in VALID_STAGES:
        raise ValueError(stage)
    if not store.get(did):
        raise ValueError(f"Unknown dataset {did}")
    with LOCK:
        if did in RUNS:
            raise RuntimeError(f"Dataset {did} already has a running or starting pipeline stage")
        RUNS[did] = _STARTING
        CANCEL_REQUESTED.discard(did)
        try:
            _write_state(did, stage=stage, pid=None, status="starting")
            threading.Thread(target=_run, args=(did, stage), daemon=True).start()
        except Exception:
            RUNS.pop(did, None)
            CANCEL_REQUESTED.discard(did)
            _clear_state(did)
            raise
    return {"started": True, "dataset_id": did, "stage": stage}


def _terminate_process_tree(p) -> None:
    """Request worker-tree termination and escalate if the leader won't exit."""
    try:
        if os.name == "posix":
            import signal
            os.killpg(p.pid, signal.SIGTERM)
        else:
            # Windows taskkill /T targets the full descendant tree, not only the leader.
            result = subprocess.run(
                ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 and p.poll() is None:
                p.terminate()
        p.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            import signal
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            subprocess.run(
                ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            if p.poll() is None:
                p.kill()
        p.wait()
    except ProcessLookupError:
        p.wait()


def stop(did):
    with LOCK:
        p = RUNS.get(did)
        if p is None:
            return False
        if p is _STARTING:
            CANCEL_REQUESTED.add(did)
            return True
        if p.poll() is not None:
            return False
        CANCEL_REQUESTED.add(did)
    _terminate_process_tree(p)
    return True


reconcile_orphaned_runs()
