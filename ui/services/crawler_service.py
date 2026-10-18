"""Backend bridge for crawler telemetry and explicit crawl controls."""
from __future__ import annotations
import subprocess
from ..core import registry
class CrawlerService:
    def processes(self) -> str:
        cmd=("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -match 'web_crawler|github_crawler|arxiv_crawler|huggingface_crawler|google_crawler|run_pipeline' } | Select-Object ProcessId,Name,WorkingSetSize,CommandLine | Format-List")
        try:
            p=subprocess.run(["powershell","-NoProfile","-Command",cmd],capture_output=True,text=True,timeout=15,check=False); out=p.stdout.strip(); return out if out else "No crawler processes detected."
        except Exception as e:return f"Process inspection unavailable: {e}"
    def start(self,did): return registry.process().post(f"/api/datasets/{did}/stage/crawl")
    def stop(self,did): return registry.process().post(f"/api/datasets/{did}/stop")
    def state(self,did): return registry.process().get(f"/api/datasets/{did}")
    def stats(self,did):
        try:
            result=registry.process().get(f"/api/datasets/{did}/crawl/stats"); return result if isinstance(result,dict) else {}
        except Exception:return {}
    def domain_signals(self,did):
        try:
            result=registry.process().get(f"/api/datasets/{did}/crawl/domains")
            if isinstance(result,list): return [(r.get("domain",""),float(r.get("avg_score",0)),int(r.get("count",0))) for r in result]
        except Exception:pass
        return []
    def log_tail(self,did,n=80):
        try:
            result=registry.process().get(f"/api/datasets/{did}/crawl/log",params={"tail":n})
            if isinstance(result,list):return [str(x) for x in result]
            if isinstance(result,str):return result.splitlines()[-n:]
        except Exception:pass
        return []
