#!/usr/bin/env python3
"""Evaluate a trained Model Lab checkpoint and write quantitative/qualitative evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any
import numpy as np
try:
    import torch
    _no_grad = torch.no_grad
except ImportError:
    torch = None  # type: ignore[assignment]
    def _no_grad():
        return lambda fn: fn
from pipeline.shardwriter.shard_writer import ShardDataLoader
from pipeline.trainer.model import LlamaModel, ModelConfig
from pipeline.trainer.train import checkpoint_metadata, select_checkpoint
from pipeline.experiment_db import ExperimentDB
ROOT = Path(__file__).resolve().parents[1]

PROMPTS = {
    "general": ["Explain why reliable software testing matters.", "Describe a practical way to solve a difficult problem."],
    "technical": ["Explain what a database transaction does.", "Describe how a neural network learns from examples."],
    "science": ["Explain why the sky appears blue during the day.", "Describe the role of energy in a physical system."],
    "history": ["Explain why primary sources matter to historians.", "Describe how technology can change societies."],
    "code": ["Write a short Python function that returns the larger of two numbers.", "Explain what a unit test should verify."],
}

def _tokenizer(path: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))

def _load_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = LlamaModel(ModelConfig.from_dict(payload["model_cfg"]))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload

def _perplexity(loss: float) -> float:
    return math.exp(min(float(loss), 20.0))

@_no_grad()
def _validation_loss(model, shard_dir: Path, seq_len: int, batch_size: int, batches: int, device) -> float:
    loader = ShardDataLoader(shard_dir, "val", seq_len, dtype=np.uint16, seed=2026)
    losses = []
    for _ in range(max(1, batches)):
        x, y = loader.next_batch(batch_size)
        _, loss = model(x.to(device), y.to(device))
        losses.append(float(loss.item()))
    return float(np.mean(losses))

def _ngrams(tokens: list[int], n: int):
    return [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]

def _generation_metrics(token_ids: list[int], text: str):
    if not token_ids:
        return {"tokens": 0, "repetition_rate": 1.0, "distinct_1": 0.0, "distinct_2": 0.0, "repeated_runs": 0}
    one, two = _ngrams(token_ids, 1), _ngrams(token_ids, 2)
    d1 = len(set(one)) / len(one) if one else 0.0
    d2 = len(set(two)) / len(two) if two else 0.0
    repeated_runs = len(re.findall(r"(\b\w+\b)(?:\s+\1){2,}", text, flags=re.IGNORECASE))
    return {"tokens": len(token_ids), "repetition_rate": round(1.0-d1, 6), "distinct_1": round(d1, 6), "distinct_2": round(d2, 6), "repeated_runs": repeated_runs}

def _qualitative(text: str, metrics: dict[str, Any]):
    language = "PASS" if text.strip() and sum(ch.isalpha() for ch in text) >= 3 else "FAIL"
    coherence = "LIMITED" if metrics["tokens"] < 12 or metrics["repetition_rate"] > 0.45 or metrics["distinct_2"] < 0.55 else "OBSERVED"
    return {"language": language, "coherence": coherence}

def _evaluate_generation(model, tokenizer, device, seed: int, max_new_tokens: int):
    domains = {}
    for domain, prompts in PROMPTS.items():
        rows = []
        for index, prompt in enumerate(prompts):
            random.seed(seed + index); np.random.seed(seed + index); torch.manual_seed(seed + index)
            encoded = tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
            output = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
            generated_ids = output[0].tolist()[len(encoded.ids):]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            metrics = _generation_metrics(generated_ids, text)
            rows.append({"prompt": prompt, "text": text, "metrics": metrics, "quality": _qualitative(text, metrics)})
        domains[domain] = rows
    return domains

def _summary(domains):
    rows = [r for values in domains.values() for r in values]
    if not rows:
        return {"language": "FAIL", "coherence": "LIMITED", "domain_behavior": "FAIL"}
    return {
        "language": "PASS" if all(r["quality"]["language"] == "PASS" for r in rows) else "FAIL",
        "coherence": "OBSERVED" if sum(r["quality"]["coherence"] == "OBSERVED" for r in rows) >= len(rows)*0.75 else "LIMITED",
        "domain_behavior": "PASS" if all(any(r["quality"]["language"] == "PASS" for r in v) for v in domains.values()) else "LIMITED",
        "mean_repetition_rate": round(float(np.mean([r["metrics"]["repetition_rate"] for r in rows])), 6),
        "mean_distinct_1": round(float(np.mean([r["metrics"]["distinct_1"] for r in rows])), 6),
        "mean_distinct_2": round(float(np.mean([r["metrics"]["distinct_2"] for r in rows])), 6),
        "prompt_count": len(rows), "domain_count": len(domains),
    }

def _markdown(report):
    m, s = report["model"], report["generation"]["summary"]
    lines = ["# MODEL EVALUATION", "",
        f"- Checkpoint: {report['checkpoint']['path']}",
        f"- Checkpoint SHA-256: {report['checkpoint']['sha256']}",
        f"- Parameters: **{m['parameters']:,}**", f"- Training steps: **{m['training_steps']:,}**", "",
        "## Held-out validation", "", f"- Validation loss: **{report['validation']['loss']:.4f}**",
        f"- Perplexity: **{report['validation']['perplexity']:.2f}**", f"- Batches: {report['validation']['batches']}", "",
        "## Generation", "", f"- Language: **{s['language']}**", f"- Coherence: **{s['coherence']}**",
        f"- Domain behavior: **{s['domain_behavior']}**", f"- Repetition rate: **{s['mean_repetition_rate']:.3f}**",
        f"- Distinct-1: **{s['mean_distinct_1']:.3f}**", f"- Distinct-2: **{s['mean_distinct_2']:.3f}**",
        f"- Domains tested: **{s['domain_count']}**", f"- Fixed prompts: **{s['prompt_count']}**", "",
        "## Checkpoint comparison", "", "| Checkpoint | Step | Val loss | Perplexity |", "|---|---:|---:|---:|"]
    for row in report["checkpoint_comparison"]:
        lines.append(f"| {row['path']} | {row['step']} | {row['val_loss']:.4f} | {row['perplexity']:.2f} |")
    lines += ["", "## Interpretation", "", "Training success is not model-quality evidence. A completed optimization run can be reproducible and operationally successful while the resulting language model remains weak.", "", "Generation outputs are qualitative evidence, not a claim of human-level semantic quality.", "", "## Fixed-prompt outputs", ""]
    for domain, rows in report["generation"]["domains"].items():
        lines.append(f"### {domain}")
        for row in rows: lines += [f"Prompt: {row['prompt']}", "", row["text"].replace("\n", " "), ""]
    return "\n".join(lines) + "\n"

def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a Model Lab checkpoint.")
    p.add_argument("--checkpoint", default="best", choices=["best", "final", "latest"])
    p.add_argument("--checkpoint-path"); p.add_argument("--output-dir", default="output"); p.add_argument("--shard-dir"); p.add_argument("--tokenizer")
    p.add_argument("--val-batches", type=int, default=20); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--max-new-tokens", type=int, default=64); p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    ckpt = Path(args.checkpoint_path).resolve() if args.checkpoint_path else select_checkpoint(output_dir / "checkpoints", args.checkpoint)
    if ckpt is None: raise SystemExit("No valid checkpoint found.")
    meta = checkpoint_metadata(ckpt)
    train_cfg = meta.get("train_cfg") or {}
    tokenizer_path = Path(args.tokenizer or train_cfg.get("tokenizer_path") or output_dir / "tokenizer" / "tokenizer.json")
    shard_dir = Path(args.shard_dir or train_cfg.get("shard_dir") or output_dir / "shards")
    if not tokenizer_path.is_file(): raise SystemExit(f"Tokenizer not found: {tokenizer_path}")
    if not shard_dir.is_dir(): raise SystemExit(f"Shard directory not found: {shard_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = _load_checkpoint(ckpt, device); tokenizer = _tokenizer(tokenizer_path)
    started = time.perf_counter()
    val_loss = _validation_loss(model, shard_dir, model.cfg.seq_len, args.batch_size, args.val_batches, device)
    domains = _evaluate_generation(model, tokenizer, device, args.seed, args.max_new_tokens)
    report = {"schema": 1, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {"path": str(ckpt), "sha256": meta.get("sha256"), "kind": payload.get("checkpoint_kind")},
        "model": {"parameters": model.param_count(), "config": model.cfg.to_dict(), "training_steps": int(payload.get("step", 0)), "best_val_loss": payload.get("best_val_loss")},
        "validation": {"loss": val_loss, "perplexity": _perplexity(val_loss), "batches": args.val_batches, "dataset": "held-out val shards"},
        "generation": {"summary": _summary(domains), "domains": domains}, "checkpoint_comparison": [], "duration_seconds": 0.0}
    for kind in ("best", "final", "latest"):
        candidate = select_checkpoint(output_dir / "checkpoints", kind)
        if candidate is None or any(row["path"] == str(candidate) for row in report["checkpoint_comparison"]): continue
        cp_meta = checkpoint_metadata(candidate); cp = torch.load(candidate, map_location="cpu", weights_only=False); loss = cp.get("val_loss", cp.get("best_val_loss"))
        if loss is None: continue
        report["checkpoint_comparison"].append({"path": str(candidate), "kind": cp.get("checkpoint_kind"), "step": int(cp.get("step", 0)), "val_loss": float(loss), "perplexity": _perplexity(float(loss)), "sha256": cp_meta.get("sha256")})
    report["duration_seconds"] = round(time.perf_counter() - started, 3)
    json_path, md_path = output_dir / "evaluations" / "model_evaluation.json", output_dir / "evaluations" / "MODEL_EVALUATION.md"
    json_path.parent.mkdir(parents=True, exist_ok=True); json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); md_path.write_text(_markdown(report), encoding="utf-8")
    report_sha = hashlib.sha256(json_path.read_bytes()).hexdigest()
    db = ExperimentDB(output_dir / "experiment.db")
    try:
        row = db.conn.execute("SELECT id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
        run_id = str(row["id"]) if row else ""
        if run_id:
            artifact_id = f"{run_id}:evaluation:model_quality"
            db.conn.execute("INSERT INTO artifacts(id,run_id,stage_name,path,sha256,size_bytes,created_at,parent_artifact_id,metadata_json) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET sha256=excluded.sha256,size_bytes=excluded.size_bytes,metadata_json=excluded.metadata_json",
                (artifact_id, run_id, "evaluate", str(json_path), report_sha, json_path.stat().st_size, report["generated_at"], None, json.dumps({"markdown_path": str(md_path), "checkpoint_sha256": meta.get("sha256")}, sort_keys=True)))
            db.conn.execute("INSERT INTO evaluations(run_id,name,status,score,step,artifact_id,details_json) VALUES(?,?,?,?,?,?,?)",
                (run_id, "model_quality", "COMPLETE", val_loss, int(payload.get("step", 0)), artifact_id, json.dumps(report, sort_keys=True)))
            db.conn.commit()
    finally: db.close()
    print(_markdown(report)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
