from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from pipeline.integrity import artifact_valid, checkpoint_manifest_valid, sha256_file

from .config import DATASETS, group_by_id, profile_by_group_id, validate_profile_catalog
from .security import MAX_INGEST_BYTES, MAX_INGEST_FILES, reject_symlink_tree

LOCK = RLock()
STAGES = ["crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export"]


def now(): return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class DatasetStore:
    def __init__(self):
        # Bind the dataset root when the store is constructed. The module-level
        # DATASETS setting is configuration, not mutable runtime state. Keeping
        # a stable root prevents background runner threads from switching roots
        # underneath an existing store when tests or an embedding application
        # temporarily replace the module setting.
        self._datasets = DATASETS
        self._datasets.mkdir(parents=True, exist_ok=True)
        validate_profile_catalog()

    def path(self, did):
        did = int(did)
        if did <= 0: raise ValueError("dataset_id must be a positive integer")
        return self._datasets / f"dataset_{did:03d}"

    def output_path(self, did): return self.path(did) / "output"
    def _ids(self):
        ids = set(int(p.name[8:]) for p in self._datasets.glob("dataset_*") if p.is_dir() and p.name[8:].isdigit())
        catalog = validate_profile_catalog()
        for did in catalog:
            if (self._datasets / f"dataset_{int(did):03d}").is_dir():
                ids.add(int(did))
        i = 1
        while (self._datasets / f"dataset_{i:03d}").is_dir():
            ids.add(i)
            i += 1
        return sorted(ids)

    def next_id(self):
        profile_ids = [int(p["dataset_id"]) for p in validate_profile_catalog().values()]
        return max([0, *self._ids(), *profile_ids]) + 1

    def get(self, did):
        # Serialize readers with atomic dataset.json replacement on Windows.
        # A concurrent read can keep the file handle open long enough for
        # os.replace() to fail with PermissionError on Windows.
        with LOCK:
            p = self.path(did) / "dataset.json"
            if not p.exists(): return None
            return json.loads(p.read_text(encoding="utf-8"))

    def list(self, limit=None, offset=0):
        ids = self._ids()
        if limit is not None: ids = ids[offset:offset + limit]
        return [self.get(i) for i in ids]

    def create(self, name, description="", group_id=None, group_config=None):
        with LOCK:
            profile = profile_by_group_id(group_id) if group_id else None
            if profile:
                did = int(profile["dataset_id"])
                existing = self.get(did)
                if existing:
                    if existing.get("group_id") != group_id:
                        raise ValueError(f"Canonical dataset ID {did} is bound to {existing.get('group_id')!r}, not {group_id!r}")
                    raise ValueError(f"Canonical dataset already exists: dataset_{did:03d}")
            else:
                did = self.next_id()
            group = group_by_id(group_id) if group_id else (group_config or {"id": f"custom_{did:03d}", "name": name, "sources": {}})
            if group_id and not group: raise ValueError(f"Unknown dataset group: {group_id}")
            root = self.path(did); root.mkdir(parents=True, exist_ok=True)
            for d in ("raw", "logs", "errors", "output", "scratch"): (root / d).mkdir(exist_ok=True)
            meta = {"id": did, "name": name.strip() or f"Dataset {did}", "description": description,
                    "group_id": group.get("id"), "group": group, "profile": profile, "status": "NEW",
                    "stages": {s: "pending" for s in STAGES}, "stats": {"files": 0, "bytes": 0, "documents": 0, "words": 0},
                    "events": [], "created_at": now(), "updated_at": now()}
            atomic_json(root / "dataset.json", meta); self.ensure_files(did); return meta

    def ensure_files(self, did):
        root = self.path(did); (root / "events.jsonl").touch(exist_ok=True); return root

    def ensure_seed_datasets(self):
        mapping = validate_profile_catalog()
        existing = {int(d["id"]): d for d in self.list() if d and d.get("id") is not None}
        for did, profile in sorted(mapping.items()):
            gid = profile["group_id"]
            if did in existing:
                if existing[did].get("group_id") != gid:
                    raise RuntimeError(f"Canonical dataset identity mismatch: dataset_{did:03d} is {existing[did].get('group_id')!r}, expected {gid!r}")
                continue
            current = self.get(did)
            if current:
                if current.get("group_id") != gid:
                    raise RuntimeError(f"Canonical dataset identity mismatch: dataset_{did:03d} is {current.get('group_id')!r}, expected {gid!r}")
                continue
            self.create(profile.get("name", gid), group_id=gid)
        return self.list()

    def update(self, did, **changes):
        with LOCK:
            d = self.get(did)
            if not d: raise KeyError(did)
            d.update(changes); d["updated_at"] = now(); atomic_json(self.path(did) / "dataset.json", d); return d

    def event(self, did, event, data=None):
        item = {"ts": now(), "event": event, "data": data or {}}
        with LOCK:
            with (self.path(did) / "events.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item

    def tail_events(self, did, limit=150):
        p = self.path(did) / "events.jsonl"
        if not p.exists(): return []
        try:
            from collections import deque
            with p.open(encoding="utf-8", errors="replace") as f: return list(deque((json.loads(x) for x in f if x.strip()), maxlen=limit))
        except (OSError, json.JSONDecodeError): return []

    def refresh_stats(self, did):
        with LOCK:
            root = self.path(did); files = bytes_ = docs = words = 0
            try:
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [d for d in dirnames if d not in (".git", ".runtime")]
                    files += len(filenames)
                    for filename in filenames:
                        try: bytes_ += os.path.getsize(os.path.join(dirpath, filename))
                        except OSError: pass
            except OSError: pass
            candidates = (root / "scratch" / "04_weighted.jsonl", root / "scratch" / "03_deduped.jsonl", root / "scratch" / "02_cleaned.jsonl", root / "scratch" / "01_crawled.jsonl")
            for c in candidates:
                if not c.exists(): continue
                try:
                    with c.open(encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if not line: continue
                            try:
                                data = json.loads(line); docs += 1; text = str(data.get("text", "")); words += len(text.split()) if text else 0
                            except (json.JSONDecodeError, ValueError): pass
                    break
                except OSError: pass
            d = self.get(did)
            if d:
                d["stats"] = {**d.get("stats", {}), "files": files, "bytes": bytes_, "documents": docs, "words": words}; d["updated_at"] = now(); atomic_json(root / "dataset.json", d)

    def ingest_path(self, did, source: Path):
        with LOCK:
            source = Path(source).resolve(strict=False)
            root = self.path(did)
            dest = root / "raw"
            if not source.exists():
                raise FileNotFoundError(source)
            if source.is_symlink():
                raise ValueError("symlink ingest sources are not permitted")
            if source == root or source == dest or source.is_relative_to(root):
                raise ValueError("ingest source cannot be inside the destination dataset")
            reject_symlink_tree(source)
            paths = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
            if len(paths) > MAX_INGEST_FILES:
                raise ValueError(f"ingest contains too many files; limit is {MAX_INGEST_FILES}")
            total_bytes = 0
            for p in paths:
                try:
                    size = p.stat().st_size
                except OSError as exc:
                    raise ValueError(f"cannot inspect ingest source: {p}") from exc
                total_bytes += size
                if total_bytes > MAX_INGEST_BYTES:
                    raise ValueError(f"ingest exceeds the {MAX_INGEST_BYTES} byte limit")
            for p in paths:
                rel = p.name if source.is_file() else p.relative_to(source).as_posix()
                target = (dest / rel).resolve(strict=False)
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError("ingest destination escapes the dataset raw directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target, follow_symlinks=False)
            stages = self.get(did)["stages"]
            stages["crawl"] = "complete"
            self.update(did, status="COLLECTED", stages=stages)
            self.refresh_stats(did)
            self.event(did, "ingest.completed", {"files": len(paths), "bytes": total_bytes})

    def _verified_stage(self, stage, paths):
        paths = [p for p in paths if p is not None]
        if not paths: return "pending"
        if stage in {"crawl", "clean", "dedup", "weight", "tokenize"}: return "complete" if all(artifact_valid(p) for p in paths) else "stale"
        if stage == "shard":
            marker = paths[0].parent / "shards.manifest.json"
            if not marker.is_file(): return "stale"
            try:
                data = json.loads(marker.read_text(encoding="utf-8")); files = data.get("files", [])
                if not files: return "corrupt"
                for item in files:
                    shard_path = marker.parent / item["name"]
                    if not shard_path.is_file() or shard_path.stat().st_size != int(item["size"]): return "corrupt"
                return "complete" if all(sha256_file(marker.parent / item["name"]) == item["sha256"] for item in files) else "corrupt"
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError): return "corrupt"
        if stage == "train": return "complete" if any(checkpoint_manifest_valid(p) for p in paths) else "stale"
        if stage == "export":
            manifest = paths[0]
            if not manifest.is_file() or manifest.stat().st_size <= 0: return "pending"
            try:
                data = json.loads(manifest.read_text(encoding="utf-8")); final = Path(data["final_gguf"])
                if not final.is_file() or final.stat().st_size != int(data["final_gguf_size"]): return "stale"
                if sha256_file(final) != data["final_gguf_sha256"]: return "corrupt"
                modelfile = Path(data["modelfile"])
                return "complete" if modelfile.is_file() and modelfile.stat().st_size > 0 else "stale"
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError): return "corrupt"
        return "pending"

    def refresh_pipeline_state(self, did):
        with LOCK:
            d = self.get(did)
            if not d: return None
            self.refresh_stats(did); d = self.get(did); root = self.path(did); out = root / "output"; scratch = root / "scratch"
            checks = {"crawl": list(scratch.glob("01_crawled*.jsonl")), "clean": [p for p in [scratch / "02_cleaned.jsonl"] if p.exists()], "dedup": [p for p in [scratch / "03_deduped.jsonl"] if p.exists()], "weight": [p for p in [scratch / "04_weighted.jsonl"] if p.exists()], "tokenize": [p for p in [out / "tokenizer" / "tokenizer.json"] if p.exists()], "shard": list((out / "shards").glob("shard_*.bin")), "train": sorted({*(out / "checkpoints").glob("ckpt_final.pt"), *(out / "checkpoints").glob("ckpt_final_*.pt")}), "export": [p for p in [out / "gguf" / "export_manifest.json"] if p.exists()]}
            states = {s: self._verified_stage(s, checks[s]) for s in STAGES}
            stages = d.get("stages")
            if not isinstance(stages, dict):
                d["status"] = "CORRUPT"
                d["stages"] = {s: "pending" for s in STAGES}
            else:
                for s, state in states.items():
                    if stages.get(s) != "running": stages[s] = state
                if any(v == "corrupt" for v in states.values()): d["status"] = "CORRUPT"
                elif any(v == "stale" for v in states.values()): d["status"] = "STALE"
                elif states["export"] == "complete": d["status"] = "COMPLETE"
                elif states["train"] == "complete": d["status"] = "TRAINED"
                elif any(v == "running" for v in stages.values()): d["status"] = "RUNNING"
                elif any(v == "complete" for v in states.values()): d["status"] = "IN_PROGRESS"
                else: d["status"] = "NEW"
            atomic_json(root / "dataset.json", d); return d

    def crawl_stats(self, did): return self._json_or_empty(self.path(did) / "scratch" / "crawl_stats.json")
    def crawl_domains(self, did): return self._json_or_empty(self.path(did) / "scratch" / "crawl_domains.json")
    def crawl_log(self, did, tail=80):
        p = self.path(did) / "logs" / "command-center.log"
        if not p.exists(): return []
        from collections import deque
        try:
            with p.open(encoding="utf-8", errors="replace") as f: return list(deque((x.rstrip() for x in f), maxlen=max(1, min(int(tail), 1000))))
        except OSError: return []
    def _json_or_empty(self, p):
        try: return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception: return {}


store = DatasetStore()
