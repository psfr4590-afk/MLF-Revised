"""
crawler.py — Crawler screen for Model Lab UI.

Displays live per-source counters, domain signal telemetry, crawl log tail,
and explicit start/stop controls supplied by the existing backend service.
"""
from . import *
import tkinter as tk
from tkinter import messagebox

_SOURCE_ICONS={"web":"🌐","github":"🐙","arxiv":"📄","huggingface":"🤗","google":"🔍"}

class CrawlerScreen(tk.Frame):
    def __init__(self,parent):
        super().__init__(parent,bg=BG)
        self.root=frame(self,"Crawler","Monitor and control all five crawl sources — web, GitHub, ArXiv, HuggingFace, Google.")
        self.root.pack(fill="both",expand=True); self._build(); self.refresh()
    def _build(self):
        bar=toolbar(self.root); bar.pack(fill="x",pady=(0,10)); button(bar,"↻  Refresh",self.refresh).pack(side="left"); button(bar,"▶  Start Crawl",self.start,"primary").pack(side="left",padx=7); button(bar,"■  Stop",self.stop,"danger").pack(side="left")
        cols=tk.Frame(self.root,bg=BG); cols.pack(fill="both",expand=True); cols.columnconfigure(0,weight=3); cols.columnconfigure(1,weight=2)
        left=tk.Frame(cols,bg=BG); right=tk.Frame(cols,bg=BG); left.grid(row=0,column=0,sticky="nsew",padx=(0,10)); right.grid(row=0,column=1,sticky="nsew")
        src_card=card(left,"Source Counters"); src_card.pack(fill="x",pady=(0,10)); self._src_rows={}
        header=tk.Frame(src_card,bg=PANEL); header.pack(fill="x",padx=14,pady=(4,2))
        for width,label in [(14,"Source"),(9,"Fetched"),(9,"Skipped"),(9,"Errors"),(9,"Abandoned"),(10,"Status")]: tk.Label(header,text=label,bg=PANEL,fg=MUTED,font=("Segoe UI",8,"bold"),width=width,anchor="w").pack(side="left")
        for source in ["web","github","arxiv","huggingface","google"]:
            row=tk.Frame(src_card,bg=PANEL); row.pack(fill="x",padx=14,pady=2); tk.Label(row,text=f"{_SOURCE_ICONS.get(source,'')} {source}",bg=PANEL,fg=TEXT,font=("Consolas",9),width=14,anchor="w").pack(side="left"); labels={}
            for key in ["fetched","skipped","errors","abandoned"]:
                lbl=tk.Label(row,text="—",bg=PANEL,fg=MUTED,font=("Consolas",9),width=9,anchor="w"); lbl.pack(side="left"); labels[key]=lbl
            status=tk.Label(row,text="idle",bg=PANEL,fg=MUTED,font=("Segoe UI",8),width=10,anchor="w"); status.pack(side="left"); labels["status"]=status; self._src_rows[source]=labels
        tk.Frame(src_card,bg=PANEL,height=6).pack()
        ext=card(left,"Extraction Stats"); ext.pack(fill="x",pady=(0,10)); self._ext_labels={}
        for key,label in [("trafilatura_used","Trafilatura"),("bs4_fallback","BS4 Fallback"),("lang_rejected","Lang Rejected"),("sitemaps_discovered","Sitemap URLs"),("issues_fetched","Issues (GH)"),("citations_enriched","Citations (ArXiv)")]:
            row=tk.Frame(ext,bg=PANEL); row.pack(fill="x",padx=14,pady=2); tk.Label(row,text=label,bg=PANEL,fg=MUTED,font=("Segoe UI",9),width=20,anchor="w").pack(side="left"); lbl=tk.Label(row,text="—",bg=PANEL,fg=TEXT,font=("Consolas",9),anchor="w"); lbl.pack(side="left"); self._ext_labels[key]=lbl
        tk.Frame(ext,bg=PANEL,height=6).pack()
        log_card=card(left,"Crawl Log (tail)"); log_card.pack(fill="both",expand=True); _,self._log_text=output(log_card); self._log_text.pack(fill="both",expand=True,padx=10,pady=10)
        dom=card(right,"Domain Signal Scores"); dom.pack(fill="x",pady=(0,10)); self._dom_frame=tk.Frame(dom,bg=PANEL); self._dom_frame.pack(fill="x",padx=14,pady=(4,10))
        proc=card(right,"Active Crawler Processes"); proc.pack(fill="both",expand=True); _,self._proc_text=output(proc); self._proc_text.pack(fill="both",expand=True,padx=10,pady=10)
    def refresh(self):
        did=selected_id()
        try: stats=registry.crawler().stats(did) if did else {}
        except Exception: stats={}
        for source,labels in self._src_rows.items():
            s=stats.get(source,{})
            for key in ["fetched","skipped","errors","abandoned"]:
                field="abandoned_domains" if key=="abandoned" else key; labels[key].config(text=str(s.get(field,"—")),fg=SUCCESS if key=="fetched" and s.get(field,0)>0 else ERROR if key=="errors" and s.get(field,0)>0 else WARNING if key in ("skipped","abandoned") and s.get(field,0)>0 else MUTED)
            labels["status"].config(text="● running" if s.get("running",False) else "○ idle",fg=ACCENT if s.get("running",False) else MUTED)
        web=stats.get("web",{})
        for key,lbl in self._ext_labels.items():
            val=web.get(key) or stats.get("github",{}).get(key) or stats.get("arxiv",{}).get(key); lbl.config(text=str(val) if val is not None else "—",fg=TEXT if val is not None else MUTED)
        for w in self._dom_frame.winfo_children(): w.destroy()
        try: domains=registry.crawler().domain_signals(did) if did else []
        except Exception: domains=[]
        if domains:
            for dom,score,count in domains[:12]:
                row=tk.Frame(self._dom_frame,bg=PANEL); row.pack(fill="x",pady=1); color=SUCCESS if score>=0.6 else WARNING if score>=0.35 else ERROR; tk.Label(row,text=dom,bg=PANEL,fg=TEXT,font=("Consolas",8),width=28,anchor="w").pack(side="left"); tk.Label(row,text=f"{score:.2f}",bg=PANEL,fg=color,font=("Consolas",8),width=6,anchor="e").pack(side="left"); tk.Label(row,text=f"({count} pg)",bg=PANEL,fg=MUTED,font=("Segoe UI",7),width=8,anchor="w").pack(side="left")
        else: tk.Label(self._dom_frame,text="No domain signal data yet.",bg=PANEL,fg=MUTED,font=("Segoe UI",9)).pack(anchor="w",pady=6)
        try: show(self._log_text,"\n".join(registry.crawler().log_tail(did,n=80)) if did else "No log output yet.")
        except Exception as e: show(self._log_text,f"Log unavailable: {e}")
        try: show(self._proc_text,registry.crawler().processes() or "No crawler processes detected.")
        except Exception as e: show(self._proc_text,f"Process inspection unavailable: {e}")
    def start(self):
        did=selected_id()
        if not did:return messagebox.showinfo("Crawler","Select a dataset in Datasets first.")
        try: registry.crawler().start(did); self.refresh()
        except Exception as e: messagebox.showerror("Crawler",str(e))
    def stop(self):
        did=selected_id()
        if not did:return messagebox.showinfo("Crawler","Select a dataset in Datasets first.")
        try: registry.crawler().stop(did); self.refresh()
        except Exception as e: messagebox.showerror("Crawler",str(e))
