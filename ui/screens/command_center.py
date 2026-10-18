from . import *
import tkinter as tk
from tkinter import messagebox


def _fmt(value, suffix=""):
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


class CommandCenterScreen(tk.Frame):
    """Connected operational view backed by persisted pipeline/SQLite evidence."""

    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.root = frame(
            self,
            "Command Center",
            "Live operational state from the pipeline and experiment SQLite store. No dashboard numbers are fabricated.",
        )
        self.root.pack(fill="both", expand=True)
        bar = toolbar(self.root)
        bar.pack(fill="x", pady=(0, 10))
        button(bar, "↻ Refresh", self.refresh).pack(side="left")
        button(bar, "System Check", self.system_check).pack(side="left", padx=6)
        button(bar, "Build Dataset", lambda: self.run_stage_sequence(("crawl", "clean", "dedup", "weight", "tokenize", "shard"))).pack(side="left", padx=6)
        button(bar, "Train", lambda: self.run_stage("train"), "primary").pack(side="left", padx=6)
        button(bar, "Resume", lambda: self.run_stage("train"), "primary").pack(side="left", padx=6)
        button(bar, "Export", lambda: self.run_stage("export")).pack(side="left", padx=6)
        button(bar, "Stop", self.stop, "danger").pack(side="left", padx=6)

        self.status = tk.Label(self.root, text="Loading…", bg=BG, fg=MUTED, font=("Segoe UI", 9))
        self.status.pack(anchor="w", pady=(0, 8))

        self.body = tk.Frame(self.root, bg=BG)
        self.body.pack(fill="both", expand=True)
        self.refresh()
        self.after(2500, self._poll)

    def _poll(self):
        self.refresh()
        self.after(2500, self._poll)

    def _clear(self):
        for widget in self.body.winfo_children():
            widget.destroy()

    def _label(self, parent, name, value, accent=TEXT):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=14, pady=2)
        tk.Label(row, text=name, bg=PANEL, fg=MUTED, width=22, anchor="w").pack(side="left")
        tk.Label(row, text=str(value), bg=PANEL, fg=accent, anchor="w").pack(side="left", fill="x", expand=True)
        return row

    def _section(self, title, columns=3):
        card_frame = card(self.body, title)
        card_frame.pack(fill="x", pady=(0, 10))
        grid = tk.Frame(card_frame, bg=PANEL)
        grid.pack(fill="x", padx=10, pady=(0, 10))
        for i in range(columns):
            grid.columnconfigure(i, weight=1)
        return card_frame, grid

    def _metric_grid(self, parent, values):
        row = len(parent.grid_slaves()) // max(len(values), 1)
        for i, (name, value, accent) in enumerate(values):
            metric(parent, name, value, accent).grid(row=row, column=i, sticky="nsew", padx=4)

    def refresh(self):
        self._clear()
        try:
            data = registry.process().get("/api/dashboard") or {}
            self.status.configure(text="Backend connected · SQLite-backed observability · refreshed automatically", fg=SUCCESS)
        except Exception as exc:
            self.status.configure(text=f"Command Center unavailable: {exc}", fg=ERROR)
            tk.Label(self.body, text="No backend telemetry available.", bg=BG, fg=MUTED, font=("Segoe UI", 12)).pack(pady=40)
            return

        run = data.get("run") or {}
        training = data.get("training") or {}
        dataset = data.get("dataset") or {}
        hardware = data.get("hardware") or {}
        provenance = data.get("provenance") or {}
        stages = data.get("stage") or []
        warnings = data.get("warnings") or []
        errors = data.get("errors") or []

        c, g = self._section("RUN")
        self._metric_grid(g, [
            ("Run ID", run.get("id", "—"), TEXT),
            ("Status", run.get("status", "—"), SUCCESS if str(run.get("status", "")).upper() in {"COMPLETE", "RUNNING"} else TEXT),
            ("Stage", next((s.get("stage_name") for s in stages if s.get("status") == "RUNNING"), "—"), ACCENT),
        ])
        self._label(c, "Dataset", dataset.get("dataset_group") or run.get("dataset_id") or "—")
        self._label(c, "Model", dataset.get("model") or "—")
        self._label(c, "Git", f"{run.get('git_branch') or '—'} · {run.get('git_sha') or '—'}")

        c, g = self._section("TRAINING")
        progress = training.get("progress")
        progress_text = f"{progress * 100:.1f}%" if isinstance(progress, (int, float)) else "—"
        self._metric_grid(g, [
            ("Step", _fmt(training.get("step")), TEXT),
            ("Total Steps", _fmt(training.get("total_steps")), TEXT),
            ("Progress", progress_text, ACCENT),
        ])
        self._metric_grid(g, [
            ("Loss", _fmt(training.get("loss")), TEXT),
            ("Validation Loss", _fmt(training.get("validation_loss")), TEXT),
            ("Best Loss", _fmt(training.get("best_loss")), SUCCESS),
        ])
        self._metric_grid(g, [
            ("Tokens/sec", _fmt(training.get("tokens_per_sec")), TEXT),
            ("ETA", _fmt(training.get("eta_seconds"), " s"), TEXT),
            ("Elapsed", _fmt(training.get("elapsed_seconds"), " s"), TEXT),
        ])

        c, g = self._section("DATASET")
        self._metric_grid(g, [
            ("Documents", f"{int(dataset['document_count']):,}" if dataset.get("document_count") is not None else "—", TEXT),
            ("Tokens", f"{int(dataset['token_count']):,}" if dataset.get("token_count") is not None else "—", TEXT),
            ("Sources", len(dataset.get("sources") or []), TEXT),
        ])
        self._metric_grid(g, [
            ("Train Tokens", f"{int(dataset['train_tokens']):,}" if dataset.get("train_tokens") is not None else "—", TEXT),
            ("Validation Tokens", f"{int(dataset['validation_tokens']):,}" if dataset.get("validation_tokens") is not None else "—", TEXT),
            ("Warnings", len(warnings), WARNING if warnings else SUCCESS),
        ])

        c, g = self._section("HARDWARE")
        self._metric_grid(g, [
            ("CPU", hardware.get("cpu_name") or "—", TEXT),
            ("RAM", _fmt(hardware.get("ram_gb"), " GB"), TEXT),
            ("GPU", hardware.get("gpu_name") or "—", TEXT),
        ])
        self._metric_grid(g, [
            ("VRAM", _fmt(hardware.get("gpu_memory_gb"), " GB"), TEXT),
            ("CUDA", hardware.get("cuda_version") or "—", TEXT),
            ("GPU Utilization", f"{hardware['gpu_utilization']:.1f}%" if isinstance(hardware.get("gpu_utilization"), (int, float)) else "Not persisted", TEXT if hardware.get("gpu_utilization") is not None else MUTED),
        ])

        c, g = self._section("PROVENANCE", 5)
        checks = [
            ("Configuration", provenance.get("configuration")),
            ("Dataset lineage", provenance.get("dataset_lineage")),
            ("Tokenizer", provenance.get("tokenizer")),
            ("Checkpoint", provenance.get("checkpoint")),
            ("Artifact integrity", provenance.get("artifact_integrity")),
        ]
        for i, (name, passed) in enumerate(checks):
            metric(g, name, "PASS" if passed else "PENDING", SUCCESS if passed else WARNING).grid(row=0, column=i, sticky="nsew", padx=4)

        c, g = self._section("EVIDENCE")
        self._metric_grid(g, [
            ("Checkpoints", len(data.get("checkpoints") or []), TEXT),
            ("Artifacts", len(data.get("artifacts") or []), TEXT),
            ("Errors", len(errors), ERROR if errors else SUCCESS),
        ])
        if errors:
            self._label(c, "Latest error", errors[0].get("message", "—"), ERROR)

        self._section_stages(stages)

    def _section_stages(self, stages):
        c = card(self.body, "PIPELINE STAGES")
        c.pack(fill="both", expand=True, pady=(0, 10))
        rows = [(s.get("stage_name", "—"), s.get("status", "—"), _fmt(s.get("duration_seconds"), " s")) for s in stages]
        table(c, [("stage", "Stage", 150), ("status", "Status", 130), ("duration", "Duration", 130)], rows)

    def _selected_dataset(self):
        did = selected_id()
        if not did:
            raise ValueError("Select a dataset before starting a pipeline operation.")
        return did

    def run_stage(self, stage):
        try:
            did = self._selected_dataset()
            result = registry.pipeline().run_stage(did, stage)
            self.status.configure(text=f"Started {stage} for dataset {did}", fg=SUCCESS)
            self.refresh()
            return result
        except Exception as exc:
            messagebox.showerror("Command Center", str(exc), parent=self)

    def run_stage_sequence(self, stages):
        try:
            did = self._selected_dataset()
            current = registry.dataset().get(did)
            states = current.get("stages", {}) if isinstance(current, dict) else {}
            next_stage = next((stage for stage in stages if states.get(stage) not in {"complete"}), None)
            if not next_stage:
                messagebox.showinfo("Build Dataset", "All dataset-build stages are already complete.", parent=self)
                return
            self.run_stage(next_stage)
        except Exception as exc:
            messagebox.showerror("Build Dataset", str(exc), parent=self)

    def stop(self):
        try:
            did = self._selected_dataset()
            registry.pipeline().stop(did)
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Command Center", str(exc), parent=self)

    def system_check(self):
        try:
            data = registry.process().get("/api/system")
            messagebox.showinfo("System Check", f"Backend: {data.get('application', '—')}\nPython: {data.get('python', '—')}\nPlatform: {data.get('platform', '—')}\nSecurity: localhost-only", parent=self)
        except Exception as exc:
            messagebox.showerror("System Check", str(exc), parent=self)
