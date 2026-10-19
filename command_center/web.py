from __future__ import annotations

from contextlib import asynccontextmanager
from contextlib import closing
import json
import math
from datetime import datetime, timezone
import platform
import sys

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from pipeline.version import get_version
from pipeline.readiness import build_readiness_report, write_readiness_report

from .runner import stop
from .config import ROOT
from pipeline.experiment_db import ExperimentDB
from .service import add, credential_delete, credential_list, credential_presets, credential_set, credential_test, groups, ingest, init, stage, status
from .store import store
from .security import MAX_REQUEST_BYTES, validate_tail_lines

LOCAL_ORIGINS = {"http://127.0.0.1", "http://localhost", "http://[::1]"}
LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "testclient", "testserver"}
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CONTROL_HEADER = "x-m2s-command-center"


def _sanitize_json_value(value):
    """Replace non-finite floats recursively so API responses remain strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(item) for item in value]
    return value


class LocalhostOnlyMiddleware(BaseHTTPMiddleware):
    """Keep the API local and require a non-forgeable browser custom header for mutations."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > MAX_REQUEST_BYTES:
                        return JSONResponse(status_code=413, content={"error": {"code": "REQUEST_TOO_LARGE", "message": "Request body exceeds the command-center limit"}})
                except ValueError:
                    return JSONResponse(status_code=400, content={"error": {"code": "INVALID_CONTENT_LENGTH", "message": "Invalid Content-Length"}})
            client_host = request.client.host if request.client else None
            if client_host not in LOCAL_CLIENT_HOSTS:
                message = "Command center is localhost-only."
                return JSONResponse(status_code=403, content={"detail": message, "error": {"code": "LOCAL_ONLY", "message": message}})
            if request.method in MUTATING_METHODS:
                origin = request.headers.get("origin")
                if origin and origin not in LOCAL_ORIGINS:
                    return JSONResponse(status_code=403, content={"error": {"code": "ORIGIN_REJECTED", "message": "Cross-origin control requests are not permitted"}})
                if request.headers.get(CONTROL_HEADER) != "1":
                    return JSONResponse(status_code=403, content={"error": {"code": "CONTROL_HEADER_REQUIRED", "message": "Command-center control header required"}})
        return await call_next(request)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init()
    yield


app = FastAPI(title="M²S Model Training Pipeline", version=get_version(), lifespan=lifespan)
if getattr(app.router, "lifespan_context", None) is not lifespan:
    app.router.lifespan_context = lifespan
app.add_middleware(LocalhostOnlyMiddleware)


class CredentialSet(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    secret: str = Field(min_length=1, max_length=65536)
    provider: str = Field(default="custom", max_length=128)
    kind: str = Field(default="token", max_length=64)
    env_var: str = Field(default="", max_length=128)
    description: str = Field(default="", max_length=4096)
    identity: str = Field(default="", max_length=4096)


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(default="", max_length=4096)
    group_id: str | None = Field(default=None, max_length=128)


class DatasetIngest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


def _safe_error(exc: Exception, code: str = "REQUEST_FAILED") -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, {"code": "NOT_FOUND", "message": "Requested resource was not found"})
    if isinstance(exc, ValueError):
        return HTTPException(400, {"code": code, "message": "Request validation failed"})
    return HTTPException(500, {"code": "INTERNAL_ERROR", "message": "Request could not be completed"})


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the single operator interface from authoritative API state."""
    return HTMLResponse("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Lab Command Center</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2b;--panel2:#182238;--line:#2a3854;--text:#eef3ff;--muted:#9eabc4;--accent:#79a7ff;--ok:#63d49b;--warn:#f2c66d;--err:#ff7373}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:#0b1020ee;border-bottom:1px solid var(--line)}.header{max-width:1500px;margin:auto;padding:16px 22px;display:flex;align-items:center;justify-content:space-between;gap:16px}
h1{font-size:22px;margin:0}.sub{color:var(--muted);font-size:12px}.badge{padding:6px 10px;border:1px solid var(--line);border-radius:999px;font-size:12px}
main{max-width:1500px;margin:auto;padding:22px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
button{border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:8px;padding:8px 12px;cursor:pointer}button:hover{border-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:12px}.card h2{font-size:14px;margin:0 0 12px}
.metric{font-size:22px;font-weight:700}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}.rows{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.row{background:var(--panel2);border-radius:8px;padding:9px}.value{margin-top:2px;word-break:break-word}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:11px;text-transform:uppercase}
.ok{color:var(--ok)}.warn{color:var(--warn)}.err{color:var(--err)}.muted{color:var(--muted)}#message{min-height:20px;color:var(--muted);margin-bottom:10px}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.rows{grid-template-columns:1fr}}@media(max-width:560px){.grid{grid-template-columns:1fr}main{padding:14px}}
</style></head>
<body>
<header><div class="header"><div><h1>M²S Model Training Pipeline</h1><div class="sub">Single local operator interface · authoritative SQLite state</div></div><div id="api" class="badge">CONNECTING</div></div></header>
<main><div class="toolbar">
<button onclick="refresh()">↻ Refresh</button><button onclick="runReadiness()">System Readiness</button><button onclick="runStage('crawl')">Run Crawl</button><button onclick="runStage('clean')">Run Clean</button><button onclick="runStage('dedup')">Run Dedup</button><button onclick="runStage('tokenize')">Run Tokenize</button><button onclick="runStage('train')">Train</button><button onclick="runStage('export')">Export</button><button onclick="stopRun()">Stop</button><a href="/docs" target="_blank"><button>API Docs</button></a>
</div><div id="message"></div>
<section class="grid" id="metrics"></section>
<section class="card"><h2>RUN</h2><div class="rows" id="run"></div></section>
<section class="card"><h2>TRAINING</h2><div class="rows" id="training"></div></section>
<section class="card"><h2>DATASET</h2><div class="rows" id="dataset"></div></section>
<section class="card"><h2>HARDWARE</h2><div class="rows" id="hardware"></div></section>
<section class="card"><h2>SYSTEM READINESS</h2><div id="readiness-summary" class="muted">Run System Readiness to inspect the current machine. This check does not start training.</div><div id="readiness-checks"></div></section>
<section class="card"><h2>CREDENTIAL PRESETS</h2><div class="rows" id="credentials"></div><div class="muted">Secrets are never displayed. Configure the four predefined slots below through the secure credential API.</div></section>
<section class="card"><h2>PROVENANCE</h2><div class="rows" id="provenance"></div></section>
<section class="card"><h2>PIPELINE STAGES</h2><div id="stages" class="muted">Loading...</div></section>
<section class="card"><h2>RECENT RUNS</h2><div id="runs" class="muted">Loading...</div></section>
</main>
<script>
function esc(v){return String(v===null||v===undefined?"—":v).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c];});}
function fmt(v,s){return v===null||v===undefined||v===""?"—":String(v)+String(s||"");}
function setRows(id,items){document.getElementById(id).innerHTML=items.map(function(x){return '<div class="row"><div class="label">'+esc(x[0])+'</div><div class="value '+(x[2]||"")+'">'+esc(x[1])+'</div></div>';}).join("");}
async function api(path,opts){opts=opts||{};opts.headers=Object.assign({"x-m2s-command-center":"1"},opts.headers||{});var r=await fetch(path,opts);if(!r.ok)throw new Error(r.status+": "+await r.text());return r.json();}
async function refresh(){
 try{
  var d=await api("/api/dashboard"),s=await api("/api/system"),r=await api("/api/runs?limit=10"),cp=await api("/api/credentials/presets");
  document.getElementById("api").textContent="ONLINE";document.getElementById("api").className="badge ok";
  document.getElementById("message").textContent="Authoritative state refreshed at "+new Date().toLocaleTimeString();
  var run=d.run||{},t=d.training||{},ds=d.dataset||{},hw=d.hardware||{},p=d.provenance||{},st=d.stage||[];
  var cards=[["Run",run.id],["Status",run.status],["Progress",typeof t.progress==="number"?(t.progress*100).toFixed(1)+"%":"—"],["API",s.application]];
  document.getElementById("metrics").innerHTML=cards.map(function(x){return '<div class="card"><div class="label">'+esc(x[0])+'</div><div class="metric">'+esc(x[1])+'</div></div>';}).join("");
  setRows("run",[["Run ID",run.id],["Dataset",ds.dataset_group||run.dataset_id],["Model",ds.model],["Git",(run.git_branch||"—")+" · "+(run.git_sha||"—")],["Status",run.status]]);
  setRows("training",[["Step",fmt(t.step)],["Total Steps",fmt(t.total_steps)],["Loss",fmt(t.loss)],["Validation Loss",fmt(t.validation_loss)],["Best Loss",fmt(t.best_loss)],["Tokens/sec",fmt(t.tokens_per_sec)],["Elapsed",fmt(t.elapsed_seconds," s")],["ETA",fmt(t.eta_seconds," s")]]);
  setRows("dataset",[["Documents",ds.document_count!=null?Number(ds.document_count).toLocaleString():"—"],["Tokens",ds.token_count!=null?Number(ds.token_count).toLocaleString():"—"],["Train Tokens",ds.train_tokens!=null?Number(ds.train_tokens).toLocaleString():"—"],["Validation Tokens",ds.validation_tokens!=null?Number(ds.validation_tokens).toLocaleString():"—"],["Sources",(ds.sources||[]).length],["Lineage",p.dataset_lineage?"PASS":"PENDING",p.dataset_lineage?"ok":"warn"]]);
  setRows("hardware",[["CPU",hw.cpu_name],["RAM",fmt(hw.ram_gb," GB")],["GPU",hw.gpu_name],["VRAM",fmt(hw.gpu_memory_gb," GB")],["CUDA",hw.cuda_version],["GPU Utilization",hw.gpu_utilization!=null?Number(hw.gpu_utilization).toFixed(1)+"%":"Not persisted"]]);
  setRows("credentials",cp.map(function(x){return [x.provider+" · "+x.kind,x.env_var+" · "+x.description,(x.configured?"CONFIGURED":"NOT SET")+" / "+(x.environment_set?"ENV SET":"ENV NOT SET"),x.configured?"ok":"warn"];}));
  setRows("provenance",[["Configuration",p.configuration?"PASS":"PENDING",p.configuration?"ok":"warn"],["Dataset lineage",p.dataset_lineage?"PASS":"PENDING",p.dataset_lineage?"ok":"warn"],["Tokenizer",p.tokenizer?"PASS":"PENDING",p.tokenizer?"ok":"warn"],["Checkpoint",p.checkpoint?"PASS":"PENDING",p.checkpoint?"ok":"warn"],["Artifact integrity",p.artifact_integrity?"PASS":"PENDING",p.artifact_integrity?"ok":"warn"],["Warnings",(d.warnings||[]).length],["Errors",(d.errors||[]).length,(d.errors||[]).length?"err":"ok"]]);
  document.getElementById("stages").innerHTML='<table><thead><tr><th>Stage</th><th>Status</th><th>Duration</th></tr></thead><tbody>'+st.map(function(x){return '<tr><td>'+esc(x.stage_name)+'</td><td>'+esc(x.status)+'</td><td>'+esc(fmt(x.duration_seconds," s"))+'</td></tr>';}).join("")+'</tbody></table>';
  document.getElementById("runs").innerHTML='<table><thead><tr><th>Run</th><th>Status</th><th>Started</th><th>Dataset</th><th>Git</th></tr></thead><tbody>'+(r.runs||[]).map(function(x){return '<tr><td>'+esc(x.id)+'</td><td>'+esc(x.status)+'</td><td>'+esc(x.started_at)+'</td><td>'+esc(x.dataset_id)+'</td><td>'+esc(x.git_sha)+'</td></tr>';}).join("")+'</tbody></table>';
 }catch(e){document.getElementById("api").textContent="OFFLINE";document.getElementById("api").className="badge err";document.getElementById("message").textContent=e.message;}
}
async function runReadiness(){
 try{
  document.getElementById("readiness-summary").textContent="Running non-destructive checks...";
  var r=await api("/api/readiness",{method:"POST"}),s=r.summary||{};
  var cls=s.status==="READY"?"ok":(s.status==="BLOCKED"?"err":"warn");
  document.getElementById("readiness-summary").innerHTML='<div class="metric '+cls+'">'+esc(s.status||"UNKNOWN")+'</div><div class="muted">'+esc(s.passed||0)+' passed · '+esc(s.warnings||0)+' warnings · '+esc(s.blockers||0)+' required blockers</div><div class="muted">Report SHA-256: '+esc(r.report_sha256||"unavailable")+'</div>';
  document.getElementById("readiness-checks").innerHTML='<table><thead><tr><th>Check</th><th>Status</th><th>Details</th><th>Action</th></tr></thead><tbody>'+(r.checks||[]).map(function(x){var cc=x.status==="PASS"?"ok":(x.status==="BLOCKED"?"err":"warn");return '<tr><td>'+esc(x.name)+'</td><td class="'+cc+'">'+esc(x.status)+'</td><td>'+esc(x.detail)+'</td><td>'+esc(x.recommendation||"—")+'</td></tr>';}).join("")+'</tbody></table>';
  document.getElementById("message").textContent="Readiness report saved to runtime/system_readiness.json";
 }catch(e){document.getElementById("readiness-summary").textContent="Readiness check failed: "+e.message;}
}
async function runStage(stage){try{var d=await api("/api/datasets"),did=d[0]&&d[0].id;if(!did)throw new Error("No dataset is available. Create or seed a dataset first.");await api("/api/datasets/"+did+"/stage/"+stage,{method:"POST"});document.getElementById("message").textContent="Requested "+stage+" for dataset "+did+".";setTimeout(refresh,500);}catch(e){document.getElementById("message").textContent=e.message;}}
async function stopRun(){try{var d=await api("/api/datasets"),did=d[0]&&d[0].id;if(!did)throw new Error("No dataset is available.");await api("/api/datasets/"+did+"/stop",{method:"POST"});refresh();}catch(e){document.getElementById("message").textContent=e.message;}}
refresh();setInterval(refresh,2500);
</script></body></html>""")

@app.get("/api/system")
def system():
    return {"application":"M²S Model Training Pipeline","version":app.version,"platform":sys.platform,"python":sys.version.split()[0],"machine":platform.machine(),"stages":list(store.STAGES) if hasattr(store,"STAGES") else ["crawl","clean","dedup","weight","tokenize","shard","train","export"],"security":{"api_access":"localhost-only","mutation_control":"custom-header"}}


@app.post("/api/readiness")
def system_readiness():
    """Run and persist a fresh, non-destructive readiness assessment."""
    try:
        report = build_readiness_report(ROOT)
        write_readiness_report(report, ROOT / "runtime" / "system_readiness.json")
        return report
    except Exception as exc:
        raise HTTPException(500, {"code": "READINESS_FAILED", "message": f"System readiness could not complete: {type(exc).__name__}"}) from exc



def _experiment_db() -> ExperimentDB:
    return ExperimentDB(ROOT / "output" / "experiment.db")


def _rows(db: ExperimentDB, query: str, params: tuple = ()) -> list[dict]:
    return [dict(row) for row in db.conn.execute(query, params).fetchall()]


def _latest_run(db: ExperimentDB) -> dict | None:
    rows = _rows(db, "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1")
    return rows[0] if rows else None


def _run_snapshot(run_id: str | None = None) -> dict:
    db = _experiment_db()
    try:
        run = db.get_run(run_id) if run_id else _latest_run(db)
        if not run:
            return {"run": None, "stage": [], "training": {}, "dataset": {}, "hardware": {}, "provenance": {}, "checkpoints": [], "artifacts": [], "warnings": [], "errors": []}
        rid = run["id"]
        dataset_rows = _rows(db, "SELECT * FROM datasets WHERE run_id=? ORDER BY id LIMIT 1", (rid,))
        dataset = dataset_rows[0] if dataset_rows else {}
        stages = _rows(db, "SELECT * FROM stages WHERE run_id=? ORDER BY id", (rid,))
        metrics = _rows(db, "SELECT step,metric_name,metric_value,metric_unit,recorded_at FROM metrics WHERE run_id=? ORDER BY step DESC,id DESC", (rid,))
        latest = {}
        for row in metrics:
            latest.setdefault(row["metric_name"], row)
        checkpoints = _rows(db, "SELECT * FROM checkpoints WHERE run_id=? ORDER BY step DESC", (rid,))
        artifacts = _rows(db, "SELECT id,stage_name,path,sha256,size_bytes,created_at FROM artifacts WHERE run_id=? ORDER BY stage_name,path", (rid,))
        configs = _rows(db, "SELECT config_type,sha256,path,snapshot_json FROM configs WHERE run_id=? ORDER BY config_type", (rid,))
        train_config = {}
        for cfg in configs:
            if cfg.get("config_type") in {"train", "training"}:
                try:
                    train_config = json.loads(cfg.get("snapshot_json") or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    train_config = {}
        sources = _rows(db, "SELECT s.id,s.kind,s.identifier,s.revision,s.license,s.status,ss.requests,ss.successes,ss.failures,ss.documents,ss.retries,ss.duration_seconds FROM sources s LEFT JOIN source_stats ss ON ss.source_id=s.id WHERE s.dataset_id=? ORDER BY s.id", (dataset["id"],)) if dataset.get("id") else []
        warnings = _rows(db, "SELECT code,message,created_at FROM warnings WHERE run_id=? ORDER BY id DESC", (rid,))
        errors = _rows(db, "SELECT code,message,exception_type,created_at FROM errors WHERE run_id=? ORDER BY id DESC", (rid,))
        runtime = _rows(db, "SELECT * FROM runtime_estimates WHERE run_id=? ORDER BY id DESC", (rid,))
        step = next((int(r["step"]) for r in metrics if r["step"] is not None), None)
        total_steps = next((int(latest[k]["metric_value"]) for k in ("total_steps", "train.total_steps") if k in latest), None)
        if total_steps is None:
            raw_total = train_config.get("total_steps")
            if raw_total is not None:
                try:
                    total_steps = int(raw_total)
                except (TypeError, ValueError):
                    total_steps = None
        progress = step / total_steps if step is not None and total_steps else None
        def metric_value(*names):
            return next((latest[n]["metric_value"] for n in names if n in latest), None)
        train_runtime = next((r for r in runtime if r["estimate_type"] == "training"), {})
        elapsed_seconds = None
        if run.get("started_at"):
            try:
                start = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(run["completed_at"].replace("Z", "+00:00")) if run.get("completed_at") else datetime.now(timezone.utc)
                elapsed_seconds = max(0.0, (end - start).total_seconds())
            except (TypeError, ValueError):
                elapsed_seconds = None
        eta_seconds = None
        if progress and progress > 0 and elapsed_seconds is not None and not run.get("completed_at"):
            eta_seconds = max(0.0, elapsed_seconds * (1.0 - progress) / progress)
        hardware_rows = _rows(db, "SELECT * FROM hardware WHERE run_id=? LIMIT 1", (rid,))
        hardware = hardware_rows[0] if hardware_rows else {}
        if hardware.get("snapshot_json"):
            try:
                snapshot = json.loads(hardware["snapshot_json"])
                if isinstance(snapshot, dict):
                    hardware["gpu_utilization"] = snapshot.get("gpu_utilization_percent", snapshot.get("gpu_utilization"))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return _sanitize_json_value({
            "run": run,
            "stage": stages,
            "training": {
                "step": step, "total_steps": total_steps, "progress": progress,
                "loss": metric_value("loss", "train.loss", "train_loss"),
                "validation_loss": metric_value("val_loss", "validation_loss", "train.val_loss"),
                "best_loss": next((c["best_val_loss"] for c in checkpoints if c["best_val_loss"] is not None), None),
                "tokens_per_sec": metric_value("tokens_per_sec", "tokens/sec", "train.tokens_per_sec"),
                "estimated_seconds": train_runtime.get("estimated_seconds"),
                "actual_seconds": train_runtime.get("actual_seconds"),
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
            },
            "dataset": {**dataset, "sources": sources, "model": train_config.get("model_preset") or train_config.get("model_name")},
            "hardware": hardware,
            "provenance": {
                "configuration": bool(configs),
                "dataset_lineage": bool(dataset.get("manifest_sha256")),
                "tokenizer": any(a["stage_name"] == "tokenize" and a["sha256"] for a in artifacts),
                "checkpoint": bool(checkpoints),
                "artifact_integrity": bool(artifacts) and all(a["sha256"] for a in artifacts),
                "configs": configs,
            },
            "checkpoints": checkpoints, "artifacts": artifacts, "warnings": warnings, "errors": errors,
        })
    finally:
        db.close()


@app.get("/api/runs")
def api_runs(limit: int = 20):
    with closing(_experiment_db()) as db:
        return {"runs": db.search_runs(limit=max(1, min(int(limit), 100)))}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    snapshot = _run_snapshot(run_id)
    if not snapshot["run"]:
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "Run not found"})
    return snapshot


@app.get("/api/dashboard")
def api_dashboard():
    return _run_snapshot()

@app.get("/api/datasets")
def datasets(): return status()


@app.post("/api/datasets")
def create(x: DatasetCreate):
    try: return add(x.name, x.description, x.group_id)
    except Exception as exc: raise _safe_error(exc) from exc


@app.get("/api/datasets/{did}")
def dataset(did: int):
    d = status(did)
    if not d: raise HTTPException(404, {"code":"NOT_FOUND","message":"Dataset not found"})
    d["events"] = store.tail_events(did)
    return d


@app.post("/api/datasets/{did}/ingest")
def ingest_dataset(did: int, x: DatasetIngest):
    try: return ingest(did, x.path)
    except Exception as exc: raise _safe_error(exc) from exc


@app.post("/api/datasets/{did}/stage/{name}")
def run_stage(did: int, name: str):
    try: return stage(did, name)
    except Exception as exc: raise _safe_error(exc) from exc


@app.post("/api/datasets/{did}/stop")
def stop_stage(did: int):
    try: return {"stopped": stop(did)}
    except Exception as exc: raise _safe_error(exc) from exc


@app.get("/api/groups")
def api_groups(): return groups()


@app.get("/api/credentials/presets")
def credential_presets_api(): return credential_presets()


@app.get("/api/credentials")
def creds(): return credential_list()


@app.post("/api/credentials")
def set_cred(x: CredentialSet):
    try: return credential_set(x.name, x.secret, x.provider, x.kind, x.env_var, x.description, x.identity)
    except Exception as exc: raise _safe_error(exc) from exc


@app.post("/api/credentials/{name}/test")
def test_cred(name: str):
    try: return credential_test(name)
    except Exception as exc: raise _safe_error(exc) from exc


@app.delete("/api/credentials/{name}")
def del_cred(name: str):
    try: return {"deleted": credential_delete(name)}
    except Exception as exc: raise _safe_error(exc) from exc


@app.get("/api/datasets/{did}/crawl/stats")
def crawl_stats(did: int): return store.crawl_stats(did)
@app.get("/api/datasets/{did}/crawl/domains")
def crawl_domains(did: int): return store.crawl_domains(did)
@app.get("/api/datasets/{did}/crawl/log")
def crawl_log(did: int, tail: int = 80): return store.crawl_log(did, validate_tail_lines(tail))
