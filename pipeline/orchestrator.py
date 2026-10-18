"""Pipeline orchestration for the Model Lab end-to-end training flow."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import logging
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Iterator

import yaml

from pipeline.config_validation import validate_config
from pipeline.integrity import artifact_valid, atomic_jsonl_write, atomic_write_text, sha256_file, write_manifest
from pipeline.types import Document
from pipeline.tokenizer.train_tokenizer import _bytelevel_alphabet_fingerprint
from pipeline.provenance import artifact_id, config_identities, new_run_id, snapshot_configs
from pipeline.run_tracking import RunTracker
from pipeline.observability import classify_source_status, source_record, stage_counts, throughput, write_dataset_report
from pipeline.crawler.source_scorer import DomainSignalTracker, SourceWeightLookup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("orchestrator")

STAGE_IMPLEMENTATIONS = {
    "crawl": ("pipeline.crawler.web_crawler", "WebCrawler"),
    "clean": ("pipeline.cleaner.cleaner", "Cleaner"),
    "dedup": ("pipeline.embedder.semantic_dedup", "SemanticDeduplicator"),
    "weight": ("pipeline.weighter.weighter", "DomainWeighter"),
    "tokenize": ("pipeline.tokenizer.train_tokenizer", "BPETokenizerTrainer"),
    "shard": ("pipeline.shardwriter.shard_writer", "ShardWriter"),
    "train": ("pipeline.trainer.train", "Trainer"),
    "export": ("scripts.export_gguf", "export_checkpoint"),
}

SOURCE_KINDS = ("web", "github", "arxiv", "huggingface", "google", "local_files")
REQUIRED_SOURCE_FIELDS = {"kind", "identifier", "revision", "license", "raw_source_sha256", "dataset_group"}


def _setup_logging(out_dir: Path, level: str = "INFO") -> list[logging.Handler]:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    owned_handlers: list[logging.Handler] = []
    if not any(getattr(h, "_model_lab_console", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler._model_lab_console = True
        root.addHandler(handler)
        owned_handlers.append(handler)
    log_file = (log_dir / "pipeline.log").resolve()
    if not any(getattr(h, "_model_lab_file", None) == str(log_file) for h in root.handlers):
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(fmt)
        handler._model_lab_file = str(log_file)
        root.addHandler(handler)
        owned_handlers.append(handler)
    return owned_handlers


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _hash_value(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _jsonl_write(docs: Iterator[Document], path: Path, kind: str = "jsonl", provenance: dict[str, Any] | None = None) -> int:
    count = atomic_jsonl_write(path, lambda: (doc.to_jsonl() for doc in docs))
    write_manifest(path, kind=kind, rows=count, provenance=provenance)
    log.info("Wrote %d docs → %s", count, path)
    return count


def _jsonl_read(path: Path) -> Iterator[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Required pipeline input does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("record is not an object")
                yield Document.from_dict(record)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Input integrity failure at {path}:{line_no}: {exc}") from exc


def _load_class(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def _implementation_sha256(stage: str) -> str:
    module_name, class_name = STAGE_IMPLEMENTATIONS[stage]
    module = importlib.import_module(module_name)
    target = getattr(module, class_name)
    source = Path(inspect.getsourcefile(target) or module.__file__ or "").resolve()
    digest = hashlib.sha256()
    digest.update(sha256_file(Path(__file__).resolve()).encode())
    if source.is_file():
        digest.update(sha256_file(source).encode())
    return digest.hexdigest()


def _project_relative_path(path: Path) -> str:
    """Return a stable project-relative path when possible, otherwise an absolute path."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _source_definition_paths(cfg: dict, root: Path) -> dict[str, Path]:
    crawl = cfg.get("crawl", {})
    paths = {
        "source_weights": root / str(crawl.get("source_weights_file", "config/source_weights.yaml")),
        "dataset_groups": root / str(crawl.get("dataset_groups_file", "config/dataset_groups.yaml")),
        "dataset_profiles": root / str(crawl.get("dataset_profiles_file", "config/dataset_profiles.yaml")),
    }
    web = crawl.get("web", {})
    if web.get("seed_urls_file"):
        paths["seed_urls"] = root / str(web["seed_urls_file"])
    return {name: path.resolve() for name, path in paths.items()}


def _source_entry(kind: str, identifier: str, dataset_group: str, revision: str | None = None, license_name: str = "unknown") -> dict[str, Any]:
    return {
        "kind": kind,
        "identifier": identifier,
        "revision": revision,
        "license": license_name,
        "raw_source_sha256": None,
        "dataset_group": dataset_group,
    }


def _manifest_sources(cfg: dict, source_paths: dict[str, Path], selected_group_id: str | None = None) -> list[dict[str, Any]]:
    crawl = cfg.get("crawl", {}) or {}
    dataset_path = source_paths.get("dataset_groups")
    if dataset_path and dataset_path.is_file():
        groups = list((yaml.safe_load(dataset_path.read_text(encoding="utf-8")) or {}).get("dataset_groups", []))
    else:
        groups = [{"id": "default", "sources": crawl.get("sources", {})}]
    if selected_group_id and selected_group_id != "all":
        groups = [group for group in groups if str(group.get("id")) == selected_group_id]
        if not groups:
            raise ValueError(f"Unknown dataset group for source manifest: {selected_group_id}")
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    global_web = crawl.get("web", {}) or {}
    seed_path = source_paths.get("seed_urls")
    global_seeds = []
    if seed_path and seed_path.is_file():
        global_seeds = [line.strip() for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    for group in groups:
        group_id = str(group.get("id", "default"))
        enabled = {**(crawl.get("sources", {}) or {}), **(group.get("sources", {}) or {})}
        for kind in SOURCE_KINDS:
            if not bool(enabled.get(kind, kind not in {"google", "local_files"})):
                continue
            section = {**(crawl.get(kind, {}) or {}), **(group.get(kind, {}) or {})}
            if kind == "web":
                seeds = section.get("seed_urls") or global_web.get("seed_urls") or global_seeds
                identifiers = [str(url) for url in seeds] or [f"group:{group_id}:web"]
            elif kind == "github":
                identifiers = [f"group:{group_id}:github"]
            elif kind == "arxiv":
                identifiers = [f"group:{group_id}:arxiv"]
            elif kind == "huggingface":
                datasets = section.get("datasets", []) or []
                identifiers = [f"{item.get('repo')}:{item.get('config', '')}:{item.get('split', 'train')}" for item in datasets if isinstance(item, dict) and item.get("repo")] or [f"group:{group_id}:huggingface"]
            elif kind == "google":
                queries = section.get("queries", []) or []
                identifiers = [str(query) for query in queries] or [f"group:{group_id}:google"]
            else:
                identifiers = [str(path) for path in (section.get("paths", []) or [])]
                identifiers = identifiers or [f"group:{group_id}:local_files"]
            for identifier in identifiers:
                key = (kind, identifier, group_id)
                if key not in seen:
                    seen.add(key)
                    sources.append(_source_entry(kind, identifier, group_id))
    return sources


def _write_source_manifest(path: Path, cfg: dict, source_paths: dict[str, Path], selected_group_id: str | None = None, retrieval_started_at: str | None = None, retrieval_completed_at: str | None = None, retrieval_sources: list[dict[str, Any]] | None = None) -> None:
    started = retrieval_started_at or datetime.now(timezone.utc).isoformat()
    files = {name: {"path": _project_relative_path(value), "sha256": _file_hash(value)} for name, value in source_paths.items()}
    definition_hash = _hash_value({name: value.get("sha256") for name, value in files.items()})
    manifest = {
        "schema": 3,
        "run_id": cfg.get("_run_id") or "standalone-source-definition",
        "dataset_group": selected_group_id or "all",
        "retrieval_started_at": started,
        "retrieval_completed_at": retrieval_completed_at,
        "source_definition_sha256": definition_hash,
        "source_definition_files": files,
        "sources": _manifest_sources(cfg, source_paths, selected_group_id),
        "retrieval": list(retrieval_sources or []),
        "rights_note": "License and usage terms must be verified before distribution; unknown values are intentional.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _source_manifest_is_valid(path: Path, expected_group_id: str | None = None, expected_source_definition_paths: dict[str, Path] | None = None) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"schema", "run_id", "dataset_group", "retrieval_started_at", "retrieval_completed_at", "source_definition_sha256", "source_definition_files", "sources", "rights_note"}
        if not isinstance(data, dict) or not required <= set(data):
            return False
        if int(data["schema"]) < 2 or not data["run_id"] or not data["retrieval_started_at"] or not data["retrieval_completed_at"] or not data["source_definition_sha256"]:
            return False
        manifest_group = str(data["dataset_group"])
        if expected_group_id is not None and manifest_group != expected_group_id:
            return False
        definition_files = data["source_definition_files"]
        if not isinstance(definition_files, dict) or not definition_files:
            return False
        for item in definition_files.values():
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                return False
            digest = item["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                return False
        if expected_source_definition_paths is not None:
            for name, current_path in expected_source_definition_paths.items():
                entry = definition_files.get(name)
                if entry is None or entry.get("sha256") != _file_hash(current_path):
                    return False
        sources = data["sources"]
        if not isinstance(sources, list) or not sources:
            return False
        for source in sources:
            if not isinstance(source, dict) or not REQUIRED_SOURCE_FIELDS <= set(source):
                return False
            if manifest_group != "all" and source.get("dataset_group") != manifest_group:
                return False
            digest = source["raw_source_sha256"]
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _load_or_create_run_id(output_dir: Path, resume: bool) -> str:
    """Reuse only an unfinished run when resume is explicitly enabled.

    A persisted run_id.txt is a pointer, not permission to overwrite a completed
    run. Fresh executions always receive a new identity, even when a previous
    run manifest is missing or malformed.
    """
    run_id_path = output_dir / "provenance" / "run_id.txt"
    persisted = run_id_path.read_text(encoding="utf-8").strip() if run_id_path.is_file() else ""
    if resume and persisted:
        manifest_path = output_dir / "runs" / persisted / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                ended = (manifest.get("timestamps") or {}).get("ended")
                terminal = manifest.get("final_status") in {"PASS", "WARN", "FAILED", "DEGRADED", "CANCELLED"}
                if ended is None and not terminal:
                    return persisted
            except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
                # Never overwrite an unreadable historical manifest.
                pass
    run_id = new_run_id()
    run_id_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_id_path, run_id + "\n")
    return run_id


class Pipeline:
    def __init__(self, config_path: str = "config/pipeline_config.yaml", dataset_id: int | None = None, resume: bool | None = None):
        requested = Path(config_path)
        if not requested.is_absolute():
            requested = PROJECT_ROOT / requested
        self._cfg_path = requested.resolve()
        self._project_root = PROJECT_ROOT
        if not self._cfg_path.is_file():
            raise FileNotFoundError(f"Pipeline config not found: {self._cfg_path}")
        self._config_sha256 = sha256_file(self._cfg_path)
        self.cfg = _load_config(self._cfg_path)
        self.cfg["_project_root"] = str(PROJECT_ROOT)
        self.cfg["_pipeline_config_path"] = str(self._cfg_path)
        self.cfg["_pipeline_config_relative_path"] = _project_relative_path(self._cfg_path)
        validate_config(self.cfg)
        self.dataset_id = dataset_id
        configured_out = Path(self.cfg["pipeline"].get("output_dir", "output"))
        configured_scratch = Path(self.cfg["pipeline"].get("scratch_dir", "scratch"))
        if dataset_id is not None:
            self._dataset_root = PROJECT_ROOT / "datasets" / f"dataset_{dataset_id:03d}"
            if not self._dataset_root.is_dir():
                raise FileNotFoundError(f"Dataset session does not exist: {self._dataset_root}")
            self._out = self._dataset_root / "output"
            self._scratch = self._dataset_root / "scratch"
        else:
            self._dataset_root = None
            self._out = (PROJECT_ROOT / configured_out).resolve()
            self._scratch = (PROJECT_ROOT / configured_scratch).resolve()
        self._out.mkdir(parents=True, exist_ok=True)
        self._scratch.mkdir(parents=True, exist_ok=True)
        self.cfg["pipeline"]["output_dir"] = str(self._out)
        self.cfg["pipeline"]["scratch_dir"] = str(self._scratch)
        self.cfg.setdefault("tokenizer", {})["output_path"] = str(self._out / "tokenizer")
        self.cfg.setdefault("shard", {})["output_dir"] = str(self._out / "shards")
        self.cfg.setdefault("train", {})["shard_dir"] = str(self._out / "shards")
        self.cfg.setdefault("export", {})["llamacpp_dir"] = str(PROJECT_ROOT / self.cfg.get("export", {}).get("llamacpp_dir", "llama.cpp"))
        self.cfg["_pipeline_config_sha256"] = self._config_sha256
        self._resume = bool(self.cfg["pipeline"].get("resume", True)) if resume is None else bool(resume)
        self.cfg.setdefault("train", {})["resume"] = self._resume
        self._weights_path = PROJECT_ROOT / self.cfg.get("crawl", {}).get("source_weights_file", "config/source_weights.yaml")
        self._dataset_groups_path = PROJECT_ROOT / self.cfg.get("crawl", {}).get("dataset_groups_file", "config/dataset_groups.yaml")
        self._clean_config_path = PROJECT_ROOT / self.cfg.get("clean", {}).get("config_file", "config/cleaner_config.yaml")
        self._source_definition_paths = _source_definition_paths(self.cfg, PROJECT_ROOT)
        self._source_definition_hashes = {name: _file_hash(path) for name, path in self._source_definition_paths.items()}
        self.cfg["_run_id"] = _load_or_create_run_id(self._out, self._resume)
        self._config_identities = config_identities(
            self.cfg,
            pipeline_sha256=self._config_sha256,
            source_definition_hashes=self._source_definition_hashes,
        )
        self.cfg["_source_definition_hashes"] = dict(self._source_definition_hashes)
        self.cfg["_config_identities"] = dict(self._config_identities)
        self._config_snapshots = snapshot_configs(
            self._out,
            self.cfg,
            self._config_identities,
            config_path=self._cfg_path,
        )
        dataset_meta = {}
        if self._dataset_root is not None:
            meta_path = self._dataset_root / "dataset.json"
            if meta_path.is_file():
                dataset_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dataset_identity = {
            "dataset_id": f"dataset_{self.dataset_id:03d}" if self.dataset_id is not None else f"dataset:{self._config_identities['dataset_config_sha256'][:16]}",
            "group_id": dataset_meta.get("group_id"),
            "catalog_profile": dataset_meta.get("profile", {}),
            "config_identity": self._config_identities.get("dataset_config_sha256"),
        }
        experiment_seed = {
            "dataset_id": dataset_identity["dataset_id"],
            "dataset_config_sha256": self._config_identities.get("dataset_config_sha256"),
            "tokenizer_config_sha256": self._config_identities.get("tokenizer_config_sha256"),
            "shard_config_sha256": self._config_identities.get("shard_config_sha256"),
            "model_config_sha256": self._config_identities.get("model_config_sha256"),
            "train_config_sha256": self._config_identities.get("train_config_sha256"),
        }
        experiment_id = "exp:" + hashlib.sha256(
            json.dumps(experiment_seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        source_identity = {
            "definition_sha256": self._config_identities.get("source_definition_sha256"),
            "definition_files": self._source_definition_hashes,
            "sources": _manifest_sources(self.cfg, self._source_definition_paths),
        }
        manifest_path = self._out / "runs" / self.cfg["_run_id"] / "run_manifest.json"
        if self._resume and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("final_status") not in {"PASS", "WARN", "FAILED", "DEGRADED"}:
                self._run_tracker = RunTracker.resume(PROJECT_ROOT, self._out, manifest_path)
            else:
                self.cfg["_run_id"] = new_run_id()
                run_id_path = self._out / "provenance" / "run_id.txt"
                atomic_write_text(run_id_path, self.cfg["_run_id"] + "\n")
                self._run_tracker = RunTracker(
                    PROJECT_ROOT, self._out, run_id=self.cfg["_run_id"],
                    dataset_id=dataset_identity["dataset_id"], experiment_id=experiment_id,
                    configuration=self._config_identities, dataset_identity=dataset_identity,
                    source_identity=source_identity,
                )
        else:
            self._run_tracker = RunTracker(
                PROJECT_ROOT, self._out, run_id=self.cfg["_run_id"],
                dataset_id=dataset_identity["dataset_id"], experiment_id=experiment_id,
                configuration=self._config_identities, dataset_identity=dataset_identity,
                source_identity=source_identity,
            )
        self._source_manifest_path = self._out.parent / "source_manifest.json"
        self._weights = SourceWeightLookup(str(self._weights_path))
        self._signals = DomainSignalTracker(self._weights.signal_gate_config())
        self._stage_metrics: dict[str, dict[str, Any]] = {}
        self._source_observability: list[dict[str, Any]] = []
        self._logging_handlers = _setup_logging(
            self._out,
            str(self.cfg["pipeline"].get("log_level", "INFO")),
        )
        log.info("Pipeline '%s' initialized | config=%s", self.cfg["pipeline"].get("name", "pipeline"), self._cfg_path)

    def _provenance(self, stage: str, input_path: Path | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": 3,
            "stage": stage,
            "run_id": self.cfg["_run_id"],
            "artifact_id": None,
            "pipeline_config_sha256": self._config_sha256,
            "config_paths": {"pipeline": _project_relative_path(self._cfg_path)},
            "config_identities": dict(self._config_identities),
            "identity_bundle_sha256": _hash_value(self._config_identities),
            "config_snapshots": dict(self._config_snapshots),
            "implementation_sha256": _implementation_sha256(stage),
        }
        if input_path is not None:
            data["input_sha256"] = sha256_file(input_path)
            input_manifest = Path(str(input_path) + ".manifest.json")
            if input_manifest.is_file():
                data["parent_artifact_ids"] = [artifact_id("manifest", sha256_file(input_manifest))]
        if extra:
            data.update(extra)
        data["artifact_id"] = artifact_id(stage, _hash_value(data))
        return data

    def _should_skip(self, path: Path, stage: str, provenance: dict[str, Any]) -> bool:
        if self._resume and artifact_valid(path, expected_provenance=provenance):
            log.info("[%s] Verified artifact integrity and provenance, skipping: %s", stage, path)
            return True
        if path.exists() and self._resume:
            log.warning("[%s] Existing artifact is stale, unverified, or corrupt; rebuilding: %s", stage, path)
        return False

    def _load_dataset_groups(self) -> list[dict]:
        if not self._dataset_groups_path.exists():
            return [{"id": "default", "name": "default", "sources": self.cfg.get("crawl", {}).get("sources", {})}]
        return list((yaml.safe_load(self._dataset_groups_path.read_text(encoding="utf-8")) or {}).get("dataset_groups", []))

    def _session_group(self) -> dict | None:
        if self.dataset_id is None:
            return None
        meta_path = self._dataset_root / "dataset.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing dataset metadata: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gid = meta.get("group_id")
        for group in self._load_dataset_groups():
            if group.get("id") == gid:
                return group
        if meta.get("group_config"):
            return meta["group_config"]
        raise ValueError(f"Dataset {self.dataset_id} references unknown group '{gid}'")

    def stage_crawl(self, only_group: str | None = None) -> Path:
        cfg = dict(self.cfg.get("crawl", {}))
        groups = self._load_dataset_groups()
        selected_group_id = only_group
        if self.dataset_id is not None:
            group = self._session_group()
            groups = [group] if group else []
            selected_group_id = str(group.get("id")) if group else None
        if only_group:
            groups = [g for g in groups if g.get("id") == only_group]
            if not groups:
                raise ValueError(f"Unknown dataset group id: {only_group}")
        out = self._scratch / (f"01_crawled__{only_group}.jsonl" if self.dataset_id is None and only_group else "01_crawled.jsonl")
        selected_group = groups[0] if len(groups) == 1 else None
        provenance = self._provenance("crawl", extra={
            "source_weights_sha256": _file_hash(self._weights_path),
            "dataset_groups_sha256": _file_hash(self._dataset_groups_path),
            "dataset_group": selected_group_id or "all",
            "dataset_group_sha256": _hash_value(selected_group) if selected_group else None,
        })
        if self._should_skip(out, "crawl", provenance) and _source_manifest_is_valid(
            self._source_manifest_path, selected_group_id or "all", self._source_definition_paths
        ):
            self._stage_metrics["crawl"] = stage_counts(None, out)
            self._stage_metrics["crawl"].update({"removed_count": 0, "removal_reasons": {}})
            return out

        retrieval_started = datetime.now(timezone.utc).isoformat()
        _write_source_manifest(
            self._source_manifest_path, self.cfg, self._source_definition_paths,
            selected_group_id=selected_group_id or "all",
            retrieval_started_at=retrieval_started,
            retrieval_completed_at=None,
        )
        source_records: list[dict[str, Any]] = []

        def run_crawler(group: dict, kind: str, definition: dict[str, Any], crawler_cfg: dict[str, Any]) -> Iterator[Document]:
            started = time.perf_counter()
            display_name = str(definition.get("display_name") or definition.get("repo") or definition.get("identifier") or kind)
            identifier = str(definition.get("identifier") or display_name)
            configured = {
                "kind": kind,
                "source_mode": "local" if kind == "local_files" else "online",
                "display_name": display_name,
                "identifier": identifier,
                "revision": definition.get("revision"),
                "license": definition.get("license", "unknown"),
                "dataset_group": str(group.get("id", "default")),
            }
            if kind == "google":
                key = os.getenv(crawler_cfg.get("google", {}).get("api_key_env", "GOOGLE_API_KEY"), "")
                cx = os.getenv(crawler_cfg.get("google", {}).get("cx_env", "GOOGLE_CX"), "")
                if not key or not cx:
                    source_records.append(source_record(configured=configured, retrieved=False, status="DISABLED", stats={}))
                    return
            module_map = {
                "web": ("pipeline.crawler.web_crawler", "WebCrawler"),
                "github": ("pipeline.crawler.github_crawler", "GitHubCrawler"),
                "arxiv": ("pipeline.crawler.arxiv_crawler", "ArxivCrawler"),
                "huggingface": ("pipeline.crawler.huggingface_crawler", "HuggingFaceCrawler"),
                "google": ("pipeline.crawler.google_crawler", "GoogleCrawler"),
                "local_files": ("pipeline.crawler.local_files_crawler", "LocalFilesCrawler"),
            }
            module, klass = module_map[kind]
            crawler = _load_class(module, klass)(crawler_cfg, self._weights, self._signals)
            yielded = 0
            try:
                for doc in crawler.crawl():
                    yielded += 1
                    doc.meta["dataset_group"] = group.get("id", "default")
                    yield doc
            finally:
                elapsed = time.perf_counter() - started
                stats = crawler.observability()
                stats["duration_seconds"] = elapsed
                stats["document_count"] = max(int(stats.get("document_count", 0)), yielded)
                status = classify_source_status(
                    enabled=True,
                    documents=int(stats["document_count"]),
                    errors=int(crawler.stats.get("errors", 0)),
                    skipped=int(crawler.stats.get("skipped", 0)),
                    rate_limited=int(stats.get("rate_limited", 0)),
                )
                source_records.append(source_record(
                    configured=configured,
                    retrieved=bool(stats["document_count"] or stats["request_count"]),
                    status=status,
                    stats=stats,
                ))

        def stream() -> Iterator[Document]:
            for group in groups:
                merged = dict(cfg)
                for key in ("web", "github", "arxiv", "huggingface", "google", "local_files"):
                    merged[key] = {**cfg.get(key, {}), **group.get(key, {})}
                merged["sources"] = {**(cfg.get("sources", {}) or {}), **(group.get("sources", {}) or {})}
                gid = group.get("id", "default")

                if bool(merged.get("sources", {}).get("web", True)):
                    seeds = list(merged.get("web", {}).get("seed_urls", []) or [])
                    for seed in seeds:
                        web_cfg = dict(merged)
                        web_cfg["web"] = dict(merged.get("web", {}))
                        web_cfg["web"]["seed_urls"] = [seed]
                        yield from run_crawler(group, "web", {
                            "identifier": str(seed),
                            "display_name": urlparse(str(seed)).netloc or str(seed),
                        }, web_cfg)
                else:
                    source_records.append(source_record(
                        configured={"kind": "web", "display_name": "web", "identifier": f"group:{gid}:web", "dataset_group": str(gid), "license": "unknown"},
                        retrieved=False, status="DISABLED", stats={},
                    ))

                for kind in ("github", "arxiv"):
                    enabled = bool(merged.get("sources", {}).get(kind, True))
                    if not enabled:
                        source_records.append(source_record(
                            configured={"kind": kind, "display_name": kind, "identifier": f"group:{gid}:{kind}", "dataset_group": str(gid), "license": "unknown"},
                            retrieved=False, status="DISABLED", stats={},
                        ))
                        continue
                    yield from run_crawler(group, kind, {"identifier": f"group:{gid}:{kind}", "display_name": kind}, merged)

                if bool(merged.get("sources", {}).get("huggingface", True)):
                    datasets = merged.get("huggingface", {}).get("datasets", []) or []
                    for item in datasets:
                        if isinstance(item, str):
                            item = {"repo": item}
                        if not isinstance(item, dict):
                            continue
                        hf_cfg = dict(merged)
                        hf_cfg["huggingface"] = dict(merged.get("huggingface", {}))
                        hf_cfg["huggingface"]["datasets"] = [item]
                        repo_name = str(item.get("repo", "huggingface"))
                        yield from run_crawler(group, "huggingface", {
                            "identifier": repo_name,
                            "display_name": repo_name,
                            "revision": item.get("revision"),
                            "license": item.get("license", "unknown"),
                        }, hf_cfg)
                else:
                    source_records.append(source_record(
                        configured={"kind": "huggingface", "display_name": "HuggingFace", "identifier": f"group:{gid}:huggingface", "dataset_group": str(gid), "license": "unknown"},
                        retrieved=False, status="DISABLED", stats={},
                    ))

                if bool(merged.get("sources", {}).get("google", False)):
                    queries = list(merged.get("google", {}).get("queries", []) or [])
                    for query in queries:
                        google_cfg = dict(merged)
                        google_cfg["google"] = dict(merged.get("google", {}))
                        google_cfg["google"]["queries"] = [query]
                        yield from run_crawler(group, "google", {
                            "identifier": str(query),
                            "display_name": "Google",
                        }, google_cfg)
                else:
                    source_records.append(source_record(
                        configured={"kind": "google", "display_name": "Google", "identifier": f"group:{gid}:google", "dataset_group": str(gid), "license": "unknown"},
                        retrieved=False, status="DISABLED", stats={},
                    ))

                if bool(merged.get("sources", {}).get("local_files", False)):
                    local_cfg = dict(merged)
                    local_cfg["local_files"] = dict(merged.get("local_files", {}))
                    paths = list(local_cfg["local_files"].get("paths", []) or [])
                    yield from run_crawler(group, "local_files", {
                        "identifier": ";".join(str(path) for path in paths) or f"group:{gid}:local_files",
                        "display_name": "Local filesystem",
                        "license": "user-provided-local-source",
                    }, local_cfg)
                else:
                    source_records.append(source_record(
                        configured={"kind": "local_files", "display_name": "Local filesystem", "identifier": f"group:{gid}:local_files", "dataset_group": str(gid), "license": "user-provided-local-source"},
                        retrieved=False, status="DISABLED", stats={},
                    ))

        count = atomic_jsonl_write(out, lambda: (doc.to_jsonl() for doc in stream()))
        if count == 0:
            self._source_observability = source_records
            raise RuntimeError(f"Crawl produced no documents: {out}")

        self._source_observability = source_records
        _write_source_manifest(
            self._source_manifest_path, self.cfg, self._source_definition_paths,
            selected_group_id=selected_group_id or "all",
            retrieval_started_at=retrieval_started,
            retrieval_completed_at=datetime.now(timezone.utc).isoformat(),
            retrieval_sources=source_records,
        )
        write_manifest(out, kind="crawl", rows=count, provenance=provenance)
        self._stage_metrics["crawl"] = stage_counts(None, out)
        self._stage_metrics["crawl"].update({"removal_reasons": {}, "source_count": len(source_records)})
        log.info("Wrote %d docs → %s", count, out)
        return out

    def stage_clean(self, in_path: Path, out_path: Path | None = None) -> Path:
        out = out_path or (self._scratch / "02_cleaned.jsonl")
        provenance = self._provenance("clean", in_path, {"clean_config_sha256": _file_hash(self._clean_config_path)})
        if self._should_skip(out, "clean", provenance):
            return out
        if not artifact_valid(in_path):
            raise RuntimeError(f"Clean input failed integrity validation: {in_path}")
        cleaner = _load_class("pipeline.cleaner.cleaner", "Cleaner")(str(self._clean_config_path))
        def cleaned() -> Iterator[Document]:
            for doc in _jsonl_read(in_path):
                result = cleaner.clean(doc.text, doc_id=doc.doc_id)
                if not result.kept:
                    continue
                doc.text = result.text
                doc.clean_action = result.action
                doc.clean_score = result.score
                doc.word_count = len(doc.text.split())
                doc.char_count = len(doc.text)
                yield doc
        _jsonl_write(cleaned(), out, kind="clean", provenance=provenance)
        self._stage_metrics["clean"] = stage_counts(in_path, out)
        self._stage_metrics["clean"]["removal_reasons"] = {
            "too_short": cleaner.stats["dropped_short"],
            "too_long": cleaner.stats["dropped_long"],
            "duplicate": cleaner.stats["dropped_dedup"],
            "refusal": cleaner.stats["dropped_refusal"],
        }
        return out

    def stage_embed_dedup(self, in_path: Path, out_path: Path | None = None) -> Path:
        out = out_path or (self._scratch / "03_deduped.jsonl")
        dedup_cfg = self.cfg.get("embed_dedup", {})
        provenance = self._provenance("dedup", in_path, {"dedup_config_sha256": _hash_value(dedup_cfg)})
        if self._should_skip(out, "dedup", provenance):
            return out
        if not artifact_valid(in_path):
            raise RuntimeError(f"Dedup input failed integrity validation: {in_path}")
        deduper = _load_class("pipeline.embedder.semantic_dedup", "SemanticDeduplicator")(dedup_cfg)
        buffer_size = int(dedup_cfg.get("buffer_size", 10000))
        def stream() -> Iterator[Document]:
            seen: set[str] = set()
            def unique_docs() -> Iterator[Document]:
                for doc in _jsonl_read(in_path):
                    key = hashlib.sha256(" ".join(doc.text.lower().split()).encode("utf-8", errors="replace")).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    yield doc
            yield from deduper.stream(unique_docs(), buffer_size=buffer_size)
        _jsonl_write(stream(), out, kind="dedup", provenance=provenance)
        self._stage_metrics["dedup"] = stage_counts(in_path, out)
        self._stage_metrics["dedup"]["removal_reasons"] = {"semantic_duplicate": int(deduper.stats.get("dropped", 0))}
        return out

    def stage_weight(self, in_path: Path, out_path: Path | None = None) -> Path:
        out = out_path or (self._scratch / "04_weighted.jsonl")
        weight_cfg = self.cfg.get("weight", {})
        weights_path = PROJECT_ROOT / weight_cfg.get("config_file", "config/source_weights.yaml")
        provenance = self._provenance("weight", in_path, {"weight_config_sha256": _hash_value(weight_cfg), "source_weights_sha256": _file_hash(weights_path)})
        if self._should_skip(out, "weight", provenance):
            return out
        if not artifact_valid(in_path):
            raise RuntimeError(f"Weight input failed integrity validation: {in_path}")
        weighter = _load_class("pipeline.weighter.weighter", "DomainWeighter")(str(weights_path), strategy=weight_cfg.get("strategy", "upsample"))
        _jsonl_write(weighter.apply(_jsonl_read(in_path)), out, kind="weight", provenance=provenance)
        self._stage_metrics["weight"] = stage_counts(in_path, out)
        self._stage_metrics["weight"]["removal_reasons"] = {}
        return out

    def stage_tokenize(self, corpus_path: Path):
        tok_cfg = self.cfg.get("tokenizer", {})
        trainer = _load_class("pipeline.tokenizer.train_tokenizer", "BPETokenizerTrainer")(tok_cfg)
        marker = Path(tok_cfg["output_path"]) / "tokenizer.json"
        provenance = self._provenance("tokenize", corpus_path, {"tokenizer_config_sha256": _hash_value(tok_cfg)})
        if self._should_skip(marker, "tokenize", provenance):
            return trainer.load()
        if not artifact_valid(corpus_path):
            raise RuntimeError(f"Tokenizer input failed integrity validation: {corpus_path}")
        tokenizer = trainer.train(corpus_path)
        write_manifest(
            marker,
            kind="tokenizer",
            provenance=provenance,
            extra={
                "vocab_size": tokenizer.get_vocab_size(),
                "bytelevel_alphabet_size": 256,
                "bytelevel_alphabet_sha256": _bytelevel_alphabet_fingerprint(),
            },
        )
        self._stage_metrics["tokenize"] = stage_counts(corpus_path, None)
        self._stage_metrics["tokenize"].update({
            "document_count": self._stage_metrics["tokenize"].get("input_document_count"),
            "removed_count": 0,
            "removal_reasons": {},
            "token_count": None,
            "token_count_status": "materialized during shard stage",
            "vocab_size": tokenizer.get_vocab_size(),
        })
        return tokenizer

    def stage_shard(self, corpus_path: Path, tokenizer) -> Path:
        Writer = _load_class("pipeline.shardwriter.shard_writer", "ShardWriter")
        shard_cfg = self.cfg["shard"]
        shard_dir = Path(shard_cfg["output_dir"])
        marker = shard_dir / "shards.manifest.json"
        tok_cfg = self.cfg.get("tokenizer", {})
        tokenizer_path = Path(tok_cfg["output_path"]) / "tokenizer.json" if tok_cfg.get("output_path") else None
        tokenizer_manifest = Path(str(tokenizer_path) + ".manifest.json") if tokenizer_path else None
        tokenizer_artifact = artifact_id("tokenizer-manifest", sha256_file(tokenizer_manifest)) if tokenizer_manifest and tokenizer_manifest.is_file() else None
        provenance = self._provenance("shard", corpus_path, {"shard_config_sha256": _hash_value(shard_cfg), "tokenizer_sha256": _file_hash(tokenizer_path), "tokenizer_vocab_size": tokenizer.get_vocab_size(), "tokenizer_artifact_id": tokenizer_artifact})
        if tokenizer_artifact:
            provenance["parent_artifact_ids"] = list(dict.fromkeys(provenance.get("parent_artifact_ids", []) + [tokenizer_artifact]))
            provenance["artifact_id"] = artifact_id("shard", _hash_value(provenance))
        if self._resume and marker.exists():
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
                files = data.get("files", [])
                valid = data.get("provenance") == provenance and bool(files)
                if valid:
                    for item in files:
                        shard_path = shard_dir / item["name"]
                        if not shard_path.is_file() or shard_path.stat().st_size != int(item["size"]) or not item.get("sha256") or sha256_file(shard_path) != item["sha256"]:
                            valid = False
                            break
                if valid:
                    log.info("[shard] Verified %d shards, hashes, and provenance, skipping", len(files))
                    return shard_dir
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                log.warning("[shard] Invalid shard manifest; rebuilding")
        shard_dir.mkdir(parents=True, exist_ok=True)
        for old in shard_dir.glob("shard_*.bin"):
            old.unlink(missing_ok=True)
        writer = Writer(shard_cfg, tokenizer)
        writer.write(corpus_path)
        paths = sorted(shard_dir.glob("shard_*.bin"))
        if not paths:
            raise RuntimeError("Shard stage produced no shard files")
        atomic_write_text(marker, json.dumps({"schema": 4, "files": [{"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)} for p in paths], "source": str(corpus_path.resolve()), "source_sha256": sha256_file(corpus_path), "provenance": provenance, "sequence_length": int(shard_cfg.get("sequence_length", 1024)), "tokenizer_vocab_size": tokenizer.get_vocab_size()}, separators=(",", ":")) + "\n")
        self._stage_metrics["shard"] = stage_counts(corpus_path, None)
        writer_stats = getattr(writer, "stats", {})
        self._stage_metrics["shard"].update({
            "document_count": int(writer_stats.get("docs_processed", self._stage_metrics["shard"].get("input_document_count", 0))),
            "removed_count": 0,
            "removal_reasons": {},
            "token_count": writer_stats.get("total_tokens"),
            "train_document_count": int(writer_stats.get("train_docs", 0)),
            "validation_document_count": int(writer_stats.get("val_docs", 0)),
            "train_shards": int(writer_stats.get("train_shards", len(paths))),
            "validation_shards": int(writer_stats.get("val_shards", 0)),
        })
        return shard_dir

    def stage_train(self) -> Path:
        import torch
        from pipeline.trainer.train import checkpoint_metadata, select_checkpoint

        train_cfg = self.cfg.get("train", {})
        if not torch.cuda.is_available() and not bool(train_cfg.get("allow_cpu_training", False)):
            raise RuntimeError("CUDA is unavailable and allow_cpu_training=false; refusing accidental CPU pretraining.")
        self.cfg["_pipeline_config_sha256"] = self._config_sha256
        Trainer = _load_class("pipeline.trainer.train", "Trainer")
        trainer = Trainer(self.cfg)
        try:
            trainer.run()
        finally:
            trainer.close()

        preflight_report = self._out / "preflight_report.json"
        if preflight_report.is_file():
            try:
                self._stage_metrics["train"] = {
                    "preflight": json.loads(preflight_report.read_text(encoding="utf-8")),
                }
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self._stage_metrics["train"] = {"preflight_report": str(preflight_report)}

        ckpt_dir = self._out / "checkpoints"
        selected = None

        for kind in ("final", "best", "latest"):
            try:
                candidate = select_checkpoint(ckpt_dir, kind)
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                log.warning("Checkpoint selection failed for %s: %s", kind, exc)
                continue

            if candidate is None:
                continue

            try:
                checkpoint_metadata(candidate)
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                log.warning("Skipping invalid %s checkpoint: %s (%s)", kind, candidate, exc)
                continue

            selected = candidate
            break

        if selected is None:
            raise RuntimeError(f"Training completed without a valid checkpoint in {ckpt_dir}")

        try:
            payload = torch.load(selected, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"Refusing unsafe checkpoint load for {selected}: {exc}") from exc

        training_metadata = payload.get("training_metadata") if isinstance(payload, dict) else {}
        self._stage_metrics["train"] = {
            **(self._stage_metrics.get("train") or {}),
            **(training_metadata if isinstance(training_metadata, dict) else {}),
            "checkpoint_step": int(payload.get("step", 0)) if isinstance(payload, dict) else None,
            "val_loss": payload.get("val_loss") if isinstance(payload, dict) else None,
            "best_val_loss": payload.get("best_val_loss") if isinstance(payload, dict) else None,
        }
        self._selected_checkpoint = selected
        return selected

    def stage_export(self) -> Path:
        exporter = _load_class("scripts.export_gguf", "export_checkpoint")
        exp = self.cfg.get("export", {})
        export_kwargs = {
            "output_dir": self._out,
            "llamacpp_dir": PROJECT_ROOT / exp.get("llamacpp_dir", "llama.cpp"),
            "quant": str(exp.get("quant", "Q4_K_M")).upper(),
            "model_name": exp.get("model_name", "model"),
        }
        selected = getattr(self, "_selected_checkpoint", None)
        if selected is not None:
            # Export exactly the checkpoint the training stage validated and selected.
            export_kwargs["checkpoint"] = selected
        result = exporter(**export_kwargs)

        if not isinstance(result, dict):
            raise RuntimeError("GGUF exporter returned no structured export result")

        final_gguf = result.get("final_gguf")
        if not final_gguf:
            raise RuntimeError("GGUF exporter did not report a final_gguf artifact")

        artifact = Path(str(final_gguf)).resolve()
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise RuntimeError(f"GGUF export artifact is missing or empty: {artifact}")

        reported_sha = result.get("final_gguf_sha256")
        if reported_sha:
            actual_sha = sha256_file(artifact)
            if actual_sha != reported_sha:
                raise RuntimeError(
                    f"GGUF export SHA-256 mismatch for {artifact}: "
                    f"reported={reported_sha}, actual={actual_sha}"
                )

        self._stage_metrics["export"] = {
            "final_gguf": str(artifact),
            "final_gguf_sha256": sha256_file(artifact),
            "final_gguf_size": artifact.stat().st_size,
            "quantization": result.get("quantization"),
            "checkpoint": result.get("checkpoint"),
            "checkpoint_step": result.get("checkpoint_step"),
        }
        return artifact

    def _run_stage(self, name: str, fn, inputs: list[Path] | None = None):
        tracked_inputs = [
            p for p in (inputs or [])
            if isinstance(p, (str, Path)) and Path(p).exists()
        ]
        self._run_tracker.start_stage(name, inputs=[Path(p) for p in tracked_inputs])
        started = time.perf_counter()
        try:
            result = fn()
            elapsed = time.perf_counter() - started
            if isinstance(result, Path):
                outputs = [result]
            elif name == "tokenize":
                tokenizer_path = Path(self.cfg.get("tokenizer", {}).get("output_path", self._out / "tokenizer")) / "tokenizer.json"
                outputs = [tokenizer_path, Path(str(tokenizer_path) + ".manifest.json")]
            else:
                outputs = []
            missing_outputs = [
                path for path in outputs
                if not path.exists() or (path.is_file() and path.stat().st_size <= 0)
            ]
            if missing_outputs:
                raise RuntimeError(
                    f"Stage {name!r} returned without materializing required outputs: "
                    + ", ".join(str(path) for path in missing_outputs)
                )
            metrics = dict(self._stage_metrics.get(name, {}))
            metrics["duration_seconds"] = round(elapsed, 6)
            metrics["throughput_docs_per_second"] = throughput(metrics.get("document_count"), elapsed)
            if name == "crawl":
                self._run_tracker.manifest["source_observability"] = list(self._source_observability)
            self._run_tracker.finish_stage(name, outputs=outputs, metrics=metrics)
            return result
        except BaseException as exc:
            self._run_tracker.fail_stage(name, exc)
            raise

    def close(self) -> None:
        """Release resources owned by this Pipeline instance."""
        handlers = getattr(self, "_logging_handlers", [])
        self._logging_handlers = []
        root = logging.getLogger()
        for handler in handlers:
            try:
                handler.flush()
            finally:
                root.removeHandler(handler)
                handler.close()

        run_tracker = getattr(self, "_run_tracker", None)
        self._run_tracker = None
        if run_tracker is not None:
            run_tracker.close()

    def run(self, stages: str = "all", dataset_group: str | None = None):
        try:
            stage_names = ["crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export"]
            if stages == "all":
                configured = self.cfg.get("stages", {}) or {}
                requested = [
                    stage for stage in stage_names
                    if bool(configured.get(stage, False))
                ]
            else:
                requested = [s.strip() for s in stages.split(",") if s.strip()]
            unknown = [s for s in requested if s not in stage_names]
            if unknown:
                raise ValueError(f"Unknown stages: {unknown}")
            if not requested:
                raise ValueError("No enabled pipeline stages are configured")

            self._run_tracker.manifest["dataset_identity"]["requested_group"] = dataset_group or "all"
            for stage in stage_names:
                self._run_tracker.manifest["stages"].setdefault(stage, {
                    "run_id": self._run_tracker.run_id, "status": "PENDING",
                    "start": None, "end": None, "duration_seconds": None,
                    "warnings": [], "errors": [], "inputs": [], "outputs": [], "artifact_hashes": {},
                })
                if stage not in requested:
                    self._run_tracker.manifest["stages"][stage]["status"] = "SKIPPED"
            self._run_tracker._write()

            try:
                artifacts: dict[str, Any] = {}
                if "crawl" in requested:
                    artifacts["crawl"] = self._run_stage("crawl", lambda: self.stage_crawl(dataset_group))
                if "clean" in requested:
                    inp = artifacts.get("crawl", self._scratch / "01_crawled.jsonl")
                    artifacts["clean"] = self._run_stage("clean", lambda: self.stage_clean(inp), [inp])
                if "dedup" in requested:
                    inp = artifacts.get("clean", self._scratch / "02_cleaned.jsonl")
                    artifacts["dedup"] = self._run_stage("dedup", lambda: self.stage_embed_dedup(inp), [inp])
                if "weight" in requested:
                    inp = artifacts.get("dedup", self._scratch / "03_deduped.jsonl")
                    artifacts["weight"] = self._run_stage("weight", lambda: self.stage_weight(inp), [inp])
                if "tokenize" in requested:
                    inp = artifacts.get("weight", self._scratch / "04_weighted.jsonl")
                    artifacts["tokenizer"] = self._run_stage("tokenize", lambda: self.stage_tokenize(inp), [inp])
                if "shard" in requested:
                    corpus = artifacts.get("weight", self._scratch / "04_weighted.jsonl")
                    tokenizer = artifacts.get("tokenizer")
                    if tokenizer is None:
                        tokenizer_cls = _load_class("pipeline.tokenizer.train_tokenizer", "BPETokenizerTrainer")
                        tokenizer = tokenizer_cls(self.cfg.get("tokenizer", {})).load()
                    artifacts["shard"] = self._run_stage("shard", lambda: self.stage_shard(corpus, tokenizer), [corpus])
                if "train" in requested:
                    inp = artifacts.get("shard", self._out / "shards")
                    artifacts["train"] = self._run_stage("train", self.stage_train, [inp])
                if "export" in requested:
                    artifacts["export"] = self._run_stage("export", self.stage_export, [artifacts.get("train")])

            except BaseException:
                self._run_tracker.finish_run("FAILED")
                raise

            statuses = [self._run_tracker.manifest["stages"][stage]["status"] for stage in requested]
            final_status = "DEGRADED" if "DEGRADED" in statuses else ("WARN" if "WARN" in statuses else "PASS")
            report_path = self._out / "runs" / self._run_tracker.run_id / "dataset_report.json"
            write_dataset_report(
                report_path,
                run_id=self._run_tracker.run_id,
                stage_metrics=dict(self._stage_metrics),
                sources=list(self._source_observability),
            )
            self._run_tracker.manifest["dataset_report"] = {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
            }
            self._run_tracker.finish_run(final_status)
            log.info("Pipeline completed stages=%s run_id=%s dataset_report=%s", requested, self._run_tracker.run_id, report_path)
            return artifacts
        finally:
            self.close()
