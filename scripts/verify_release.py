#!/usr/bin/env python3
"""Deterministic production verification for Model Lab.

The native gate runs a bounded, network-free release fixture before native inference.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import platform
import subprocess
import sys
import sqlite3
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

NATIVE_EVIDENCE: dict[str, object] = {}
SOURCE_MANIFEST_PATH: Path | None = None
SOURCE_RUN_MANIFEST_PATH: Path | None = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402
from pipeline.integrity import write_manifest  # noqa: E402
from pipeline.tokenizer.special_tokens import special_token_ids  # noqa: E402
from pipeline.provenance import stable_hash  # noqa: E402
from pipeline.run_tracking import _git, _git_state  # noqa: E402
from pipeline.types import Document  # noqa: E402

LINT_TARGETS = ["command_center", "pipeline", "tests", "scripts", "run_pipeline.py", "run_command_center.py", "launch.py", "bootstrap.py"]
RELEASE_SMOKE_SOURCES = (("local-technical", "systems architecture, compilers, operating systems, networking, storage, and distributed computing"),("local-scientific", "physics, chemistry, biology, astronomy, mathematics, statistics, and experimental methods"),("local-engineering", "mechanical, electrical, civil, software, control systems, reliability, and manufacturing engineering"),("local-humanities", "history, literature, linguistics, philosophy, archaeology, anthropology, and cultural studies"),("local-business", "accounting, finance, economics, operations, logistics, management, markets, and entrepreneurship"),("local-geography", "cartography, climate, geology, ecology, oceans, weather, agriculture, and geographic information"),("local-medicine", "anatomy, physiology, epidemiology, diagnostics, pharmacology, public health, and clinical research"),("local-creative", "music, visual design, architecture, photography, theater, film, animation, and digital media"))


def run(cmd: list[str]) -> int:
    print("$", " ".join(map(str, cmd))); return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _write_local_smoke_input(scratch: Path, pipeline_config_sha256: str | None = None, run_id: str | None = None, config_identities: dict | None = None) -> None:
    scratch.mkdir(parents=True, exist_ok=True); corpus = scratch / "04_weighted.jsonl"; docs = []
    for source_index, (source, topic_text) in enumerate(RELEASE_SMOKE_SOURCES):
        for local_index in range(32):
            index = source_index * 32 + local_index; unique_terms = " ".join(f"{source.replace('-', '')}term{index:03d}_{suffix}" for suffix in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")); text = f"Deterministic {source} release fixture document {index:03d}. This local source covers {topic_text}. The verification corpus includes {unique_terms}. The fixture exercises tokenization, sharding, training, GGUF export, artifact integrity, and native inference without depending on the public Internet."
            docs.append(Document(doc_id=f"release-smoke-{index:03d}", url=f"file://release-smoke/{source}/{local_index}", source=source, title=f"Deterministic {source} release fixture", text=text, language="en", content_type="text/plain", domain=f"{source}.release.fixture", final_weight=1.0, meta={"release_fixture": True, "source_family": source, "source_index": source_index}))
    corpus.write_text("".join(json.dumps(doc.to_jsonl(), sort_keys=True) + "\n" for doc in docs), encoding="utf-8")
    provenance = {"schema": 2, "stage": "weight", "fixture": "local-release-multi-source", "source_families": len(RELEASE_SMOKE_SOURCES)}
    if pipeline_config_sha256: provenance["pipeline_config_sha256"] = pipeline_config_sha256
    if run_id: provenance["run_id"] = run_id
    if config_identities:
        provenance["config_identities"] = dict(config_identities)
        provenance["identity_bundle_sha256"] = stable_hash(config_identities)
    write_manifest(corpus, kind="weight", rows=len(docs), provenance=provenance)


def _write_smoke_source_manifest(tmp_root: Path, run_id: str, source_definition_paths: dict[str, Path]) -> None:
    path = tmp_root / "source_manifest.json"
    now = datetime.now(timezone.utc).isoformat()
    files = {name: {"path": str(value.relative_to(ROOT)).replace("\\\\", "/"), "sha256": _sha256(value)} for name, value in source_definition_paths.items()}; source_definition_sha256 = __import__("hashlib").sha256(json.dumps({name: value["sha256"] for name, value in files.items()}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    path.write_text(json.dumps({"schema": 2, "run_id": run_id, "dataset_group": "release-smoke", "retrieval_started_at": now, "retrieval_completed_at": now, "source_definition_sha256": source_definition_sha256, "source_definition_files": files, "sources": [{"kind": "local_fixture", "identifier": source, "revision": "embedded", "license": "project-test-fixture", "raw_source_sha256": None, "dataset_group": "release-smoke"} for source, _ in RELEASE_SMOKE_SOURCES], "rights_note": "Deterministic test fixtures only; not external training sources."}, indent=2) + "\n", encoding="utf-8")


def _validate_source_manifest(path: Path) -> None:
    required_top_level = {"schema", "run_id", "dataset_group", "retrieval_started_at", "retrieval_completed_at", "source_definition_sha256", "source_definition_files", "sources", "rights_note"}; data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not required_top_level <= set(data): raise RuntimeError("Source manifest is missing required top-level metadata")
    if int(data["schema"]) != 2 or not isinstance(data["sources"], list) or not data["sources"]: raise RuntimeError("Source manifest schema is invalid")
    required_source = {"kind", "identifier", "revision", "license", "raw_source_sha256", "dataset_group"}
    for source in data["sources"]:
        if not isinstance(source, dict) or not required_source <= set(source): raise RuntimeError("Source manifest source entry is incomplete")
        if source.get("dataset_group") != data["dataset_group"]: raise RuntimeError("Source manifest dataset-group identity is invalid")
        digest = source["raw_source_sha256"]
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower())): raise RuntimeError("Source manifest raw_source_sha256 is invalid")


def _validate_export_cards(manifest: dict[str, object]) -> None:
    """Require both export cards to exist, contain data, and match their recorded hashes."""
    for key in ("dataset_card", "model_card"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Export manifest is missing card path: {key}")
        card = Path(value)
        if not card.is_file() or card.stat().st_size <= 0:
            raise RuntimeError(f"Missing or empty export card: {key}: {card}")
        expected_hash = manifest.get(f"{key}_sha256")
        if not isinstance(expected_hash, str) or expected_hash != _sha256(card):
            raise RuntimeError(f"Export card hash missing or mismatched: {key}: {card}")


def _source_retrieval_smoke() -> bool:
    """Exercise crawl -> clean -> dedup -> weight using a deterministic offline crawler."""
    import yaml
    import pipeline.orchestrator as orchestrator

    source = ROOT / "config" / "pipeline_config.smoke.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    with tempfile.TemporaryDirectory(prefix="model-lab-source-smoke-") as tmp:
        tmp_root = Path(tmp)
        output = tmp_root / "output"
        scratch = tmp_root / "scratch"
        output.mkdir()
        scratch.mkdir()
        groups_path = tmp_root / "dataset_groups.yaml"
        groups_path.write_text(
            yaml.safe_dump({
                "dataset_groups": [{
                    "id": "release-source-smoke",
                    "name": "Offline source retrieval verification",
                    "sources": {
                        "web": True, "github": False, "arxiv": False,
                        "huggingface": False, "google": False,
                    },
                    "web": {"seed_urls": ["https://example.invalid/release-smoke"]},
                }]
            }, sort_keys=False),
            encoding="utf-8",
        )
        config.setdefault("pipeline", {})["output_dir"] = str(output)
        config["pipeline"]["scratch_dir"] = str(scratch)
        config["pipeline"]["resume"] = False
        config["stages"] = {
            "crawl": True, "clean": True, "dedup": True, "weight": True,
            "tokenize": False, "shard": False, "train": False, "export": False,
        }
        config.setdefault("crawl", {})["dataset_groups_file"] = str(groups_path)
        config["crawl"]["sources"] = {
            "web": True, "github": False, "arxiv": False,
            "huggingface": False, "google": False,
        }
        config["crawl"]["web"] = {"seed_urls": ["https://example.invalid/release-smoke"]}
        from pipeline.embedder.semantic_dedup import FAISS_AVAILABLE, ST_AVAILABLE
        if not (ST_AVAILABLE and FAISS_AVAILABLE):
            config.setdefault("embed_dedup", {})["mode"] = "fallback"
        config_path = tmp_root / "source_smoke.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        base_text = (
            "This deterministic offline source retrieval fixture exercises the production "
            "pipeline crawl, clean, deduplication, and weighting stages. It contains "
            "technical details about data validation, artifact integrity, reproducible "
            "processing, configuration, and provenance tracking. The fixture is deliberately "
            "long enough to pass the configured cleaner and does not require network access."
        )

        class OfflineCrawler:
            def __init__(self, _config, _weights, _signals):
                self.stats = {"errors": 0, "skipped": 0}

            def crawl(self):
                from pipeline.types import Document
                yield Document(
                    doc_id="release-source-1",
                    url="https://example.invalid/release-smoke/1",
                    source="web",
                    title="Offline source fixture one",
                    text=base_text,
                    meta={},
                )
                yield Document(
                    doc_id="release-source-2",
                    url="https://example.invalid/release-smoke/2",
                    source="web",
                    title="Offline source fixture two",
                    text=base_text + " Additional unique material describes corpus weighting and dataset provenance.",
                    meta={},
                )

            def observability(self):
                return {
                    "request_count": 1, "document_count": 2, "rate_limited": 0,
                    "revisions": [], "licenses": ["project-test-fixture"],
                    "source_hash": "offline-release-fixture",
                }

        original_load_class = orchestrator._load_class

        def offline_load_class(module, name):
            if module == "pipeline.crawler.web_crawler" and name == "WebCrawler":
                return OfflineCrawler
            return original_load_class(module, name)

        orchestrator._load_class = offline_load_class
        pipeline = None
        try:
            from pipeline.orchestrator import Pipeline, _source_manifest_is_valid
            pipeline = Pipeline(str(config_path), resume=False)
            crawled = pipeline.stage_crawl()
            cleaned = pipeline.stage_clean(crawled)
            deduped = pipeline.stage_embed_dedup(cleaned)
            weighted = pipeline.stage_weight(deduped)

            if not _source_manifest_is_valid(pipeline._source_manifest_path):
                raise RuntimeError("Source smoke source manifest failed validation")
            source_manifest = json.loads(
                pipeline._source_manifest_path.read_text(encoding="utf-8")
            )
            if not source_manifest.get("sources"):
                raise RuntimeError("Source smoke source manifest contains no source definitions")
            if not source_manifest.get("retrieval_completed_at"):
                raise RuntimeError("Source smoke source manifest lacks retrieval completion evidence")

            from pipeline.integrity import artifact_valid
            artifacts = [
                (crawled, "crawl"), (cleaned, "clean"),
                (deduped, "dedup"), (weighted, "weight"),
            ]
            for path, stage in artifacts:
                manifest_path = path.with_name(path.name + ".manifest.json")
                if not path.is_file() or not manifest_path.is_file():
                    raise RuntimeError(f"Source smoke missing {stage} artifact or manifest")
                if not artifact_valid(path):
                    raise RuntimeError(f"Source smoke {stage} artifact failed integrity validation")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("rows", 0) < 1:
                    raise RuntimeError(f"Source smoke {stage} artifact contains no rows")
                provenance = manifest.get("provenance", {})
                if provenance.get("stage") != stage:
                    raise RuntimeError(f"Source smoke {stage} manifest has incorrect stage provenance")

            records = [
                json.loads(line)
                for line in weighted.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not records or not all(
                isinstance(record.get("final_weight"), (int, float))
                and not isinstance(record.get("final_weight"), bool)
                for record in records
            ):
                raise RuntimeError("Source smoke weighted output lacks final_weight values")
            return True
        except Exception as exc:
            print(f"Source retrieval smoke failed: {exc}")
            return False
        finally:
            if pipeline is not None:
                pipeline.close()
            orchestrator._load_class = original_load_class



def _record_source_pipeline_integration_smoke() -> bool:
    """Record the offline upstream-stage smoke without claiming live source retrieval."""
    passed = _source_retrieval_smoke()
    NATIVE_EVIDENCE["source_pipeline_integration"] = passed
    return passed


def _native_smoke(verify_ollama: bool = False) -> int:
    source = ROOT / "config" / "pipeline_config.smoke.yaml"; config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}; config["stages"] = {"crawl": False, "clean": False, "dedup": False, "weight": False, "tokenize": True, "shard": True, "train": True, "export": True}
    if not _record_source_pipeline_integration_smoke():
        return 2
    with tempfile.TemporaryDirectory(prefix="model-lab-release-") as tmp:
        tmp_root = Path(tmp); output = tmp_root / "output"; scratch = tmp_root / "scratch"; config.setdefault("pipeline", {})["output_dir"] = str(output); config["pipeline"]["scratch_dir"] = str(scratch); config["pipeline"]["resume"] = False; config.setdefault("export", {})["llamacpp_dir"] = str(ROOT / "third_party" / "llama.cpp"); config["export"]["quant"] = "ALL"; config_path = tmp_root / "release_smoke.yaml"; config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8"); from pipeline.orchestrator import Pipeline
        smoke_pipeline = Pipeline(str(config_path), resume=False)
        run_id = smoke_pipeline.cfg["_run_id"]
        try:
            _write_local_smoke_input(
                scratch,
                pipeline_config_sha256=_sha256(config_path),
                run_id=smoke_pipeline.cfg["_run_id"],
                config_identities=smoke_pipeline._config_identities,
            )
            _write_smoke_source_manifest(
                tmp_root,
                smoke_pipeline.cfg["_run_id"],
                smoke_pipeline._source_definition_paths,
            )
        finally:
            smoke_pipeline.close()
        try: _validate_source_manifest(tmp_root / "source_manifest.json")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc: print(f"Invalid smoke source manifest: {exc}"); return 2
        if run([sys.executable, "run_pipeline.py", "--config", str(config_path), "--no-resume"]): return 2
        manifest_path = output / "gguf" / "export_manifest.json"
        if not manifest_path.is_file():
            print(f"Missing export manifest: {manifest_path}")
            return 2
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = manifest.get("artifacts") or {}
            required_quants = {"F16", "Q4_K_M", "Q5_K_M", "Q8_0"}
            if set(artifacts) != required_quants:
                print(f"Export did not produce the required quantizations: {sorted(artifacts)}")
                return 2
            for quant, record in artifacts.items():
                final_gguf = Path(record["artifact"]["path"])
                artifact_manifest = Path(record["manifest"])
                if not final_gguf.is_file() or final_gguf.stat().st_size <= 0:
                    print(f"Missing final GGUF artifact for {quant}: {final_gguf}")
                    return 2
                if _sha256(final_gguf) != record["artifact"]["sha256"]:
                    print(f"GGUF hash mismatch for {quant}: {final_gguf}")
                    return 2
                if not artifact_manifest.is_file():
                    print(f"Missing per-export manifest for {quant}: {artifact_manifest}")
                    return 2
                per_export = json.loads(artifact_manifest.read_text(encoding="utf-8"))
                required = {"checkpoint_sha256", "model_config", "export_config", "exporter_version", "timestamp", "lineage", "artifact"}
                if not required <= set(per_export):
                    print(f"Export manifest is incomplete for {quant}: {artifact_manifest}")
                    return 2
                if per_export["artifact"].get("sha256") != _sha256(final_gguf):
                    print(f"Per-export artifact hash mismatch for {quant}")
                    return 2
                _validate_export_cards(manifest)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"Invalid export manifest: {exc}")
            return 2
        for quant, record in artifacts.items():
            artifact_manifest = Path(record["manifest"])
            final_gguf = Path(record["artifact"]["path"])
            model_cfg = manifest.get("model_config") or {}
            try:
                special_ids = special_token_ids(output / "tokenizer")
            except RuntimeError as exc:
                print(f"Unable to resolve tokenizer special-token IDs: {exc}")
                return 2
            cmd = [
                sys.executable, "scripts/verify_gguf.py",
                "--model", str(final_gguf),
                "--llamacpp-dir", str(ROOT / "third_party" / "llama.cpp"),
                "--manifest", str(artifact_manifest),
                "--context-length", str(int(model_cfg.get("seq_len", 0))),
                "--bos-token-id", str(special_ids["bos"]),
                "--eos-token-id", str(special_ids["eos"]),
                "--pad-token-id", str(special_ids["pad"]),
            ]
            if run(cmd):
                print(f"GGUF inference verification failed for {quant}")
                return 2
            try:
                result = json.loads(artifact_manifest.read_text(encoding="utf-8")).get("inference_validation", {})
                if result.get("status") != "passed" or not result.get("generated_output"):
                    print(f"Export manifest does not contain a passing inference validation result for {quant}")
                    return 2
            except (OSError, ValueError, TypeError) as exc:
                print(f"Unable to read validated export manifest for {quant}: {exc}")
                return 2
        if verify_ollama:
            for quant, record in artifacts.items():
                final_gguf = Path(record["artifact"]["path"])
                modelfile = Path(record["modelfile"])
                model_name = f"model-lab-rc-{quant.lower()}"
                if run([
                    sys.executable, "scripts/verify_ollama.py",
                    "--model", str(final_gguf),
                    "--modelfile", str(modelfile),
                    "--model-name", model_name,
                ]):
                    print(f"Ollama deployment verification failed for {quant}")
                    return 2
        run_manifest_path = output / "runs" / run_id / "run_manifest.json"
        NATIVE_EVIDENCE["export"] = True
        NATIVE_EVIDENCE["inference"] = True
        _harvest_native_evidence(
            output=output,
            run_id=run_id,
            run_manifest_path=run_manifest_path,
        )
    return 0



def _is_finite_number(value: object) -> bool:
    """Accept finite numeric metrics, excluding booleans and non-numeric JSON values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _is_sha256(value: object) -> bool:
    """Return whether value is exactly a hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )



def _is_nonnegative_int(value: object) -> bool:
    """Accept non-negative integers, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_artifact_id(value: object) -> bool:
    """Validate the kind:digest shape emitted by pipeline.provenance.artifact_id."""
    if not isinstance(value, str):
        return False
    kind, separator, digest = value.partition(":")
    return bool(kind and separator and ":" not in digest and _is_sha256(digest))


def _lineage_evidence_is_valid(provenance: object, run_id: str) -> bool:
    """Require well-formed artifact identities for every required lineage link."""
    if not isinstance(provenance, dict) or provenance.get("run_id") != run_id:
        return False
    parent_ids = provenance.get("parent_artifact_ids")
    return (
        isinstance(parent_ids, list)
        and bool(parent_ids)
        and all(_is_artifact_id(parent_id) for parent_id in parent_ids)
        and _is_artifact_id(provenance.get("dataset_artifact_id"))
        and _is_artifact_id(provenance.get("tokenizer_artifact_id"))
    )


def _dataset_evidence_is_valid(
    dataset_payload: object,
    dataset_identity: object,
    metrics: object,
) -> bool:
    """Require structured, non-empty dataset evidence rather than field presence alone."""
    return (
        isinstance(dataset_payload, dict)
        and bool(dataset_payload)
        and isinstance(dataset_identity, dict)
        and bool(dataset_identity)
        and isinstance(metrics, dict)
        and bool(metrics)
    )


def _evaluation_evidence_is_valid(train_metrics: object) -> bool:
    """Require both recorded validation losses to be finite numeric values."""
    return (
        isinstance(train_metrics, dict)
        and _is_finite_number(train_metrics.get("val_loss"))
        and _is_finite_number(train_metrics.get("best_val_loss"))
    )


def _reproducibility_evidence_is_valid(train_metrics: object) -> bool:
    """Require an integer seed and a syntactically valid random-state digest."""
    if not isinstance(train_metrics, dict):
        return False
    seed = train_metrics.get("seed")
    return (
        isinstance(seed, int)
        and not isinstance(seed, bool)
        and _is_sha256(train_metrics.get("random_state_sha256"))
    )


def _validate_run_git_identity(
    *, manifest: dict, database_git_sha: object, expected_git_sha: str | None
) -> None:
    """Ensure manifest and SQLite run identity match the source being verified."""
    git_metadata = manifest.get("git")
    manifest_git_sha = git_metadata.get("commit") if isinstance(git_metadata, dict) else None
    if not isinstance(expected_git_sha, str) or not expected_git_sha:
        raise RuntimeError("current repository commit is unavailable for run identity validation")
    if manifest_git_sha != expected_git_sha:
        raise RuntimeError(
            f"run manifest git SHA mismatch: expected={expected_git_sha!r} actual={manifest_git_sha!r}"
        )
    if database_git_sha != manifest_git_sha:
        raise RuntimeError(
            f"SQLite git SHA mismatch: manifest={manifest_git_sha!r} database={database_git_sha!r}"
        )


def _auto_sizing_evidence(train_metrics: object) -> bool | None:
    """Read auto-sizing evidence from persisted train metrics."""
    if not isinstance(train_metrics, dict):
        return None

    # Current run manifests persist these fields directly under metrics.train.
    effective = train_metrics.get("effective_configuration")
    if isinstance(effective, dict):
        value = effective.get("auto_size")
        if isinstance(value, bool):
            return value

    # Also support trainer metadata wrappers used by other artifact formats.
    training_metadata = train_metrics.get("training_metadata")
    if isinstance(training_metadata, dict):
        effective = training_metadata.get("effective_configuration")
        if isinstance(effective, dict):
            value = effective.get("auto_size")
            if isinstance(value, bool):
                return value

        value = training_metadata.get("auto_size")
        if isinstance(value, bool):
            return value

    # Backward compatibility with older run manifests.
    value = train_metrics.get("auto_size")
    return value if isinstance(value, bool) else None



def _validate_final_checkpoint_metadata(meta: object) -> None:
    """Enforce the metadata contract for the final checkpoint artifact."""
    if not isinstance(meta, dict):
        raise RuntimeError("final checkpoint metadata is invalid")
    if meta.get("checkpoint_kind") != "final":
        raise RuntimeError("final checkpoint manifest is not marked as final")
    if not _is_nonnegative_int(meta.get("step")):
        raise RuntimeError("final checkpoint step metadata is invalid")


def _validate_best_checkpoint_metadata(meta: dict) -> None:
    """Enforce the checkpoint manifest contract used by best_checkpoint()."""
    if not isinstance(meta, dict):
        raise RuntimeError("best checkpoint metadata is invalid")
    if meta.get("checkpoint_kind") != "best":
        raise RuntimeError("checkpoint manifest is not marked as best")
    step = meta.get("step")
    best_step = meta.get("best_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError("best checkpoint step metadata is invalid")
    if isinstance(best_step, bool) or not isinstance(best_step, int) or best_step < 0:
        raise RuntimeError("best checkpoint best_step metadata is invalid")
    loss = meta.get("best_val_loss")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
    ):
        raise RuntimeError("best checkpoint best_val_loss must be finite")


def _validate_checkpoint_identity_chain(expected: dict, actual: dict) -> None:
    """Require stable cross-stage SHA-256 identities and exact agreement."""
    required = (
        "dataset_config_sha256",
        "tokenizer_config_sha256",
        "shard_config_sha256",
        "export_config_sha256",
        "source_definition_sha256",
    )

    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise RuntimeError("configuration identity chain is missing")

    for key in required:
        expected_hash = expected.get(key)
        actual_hash = actual.get(key)
        for label, value in (("run configuration", expected_hash),
                             ("checkpoint provenance", actual_hash)):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in value)
            ):
                raise RuntimeError(f"invalid or missing {key} in {label}")
        if actual_hash.lower() != expected_hash.lower():
            raise RuntimeError(f"checkpoint provenance identity mismatch: {key}")


def _live_source_retrieval_evidence_is_valid(
    *,
    source_manifest_path: Path,
    expected_run_id: str | None = None,
    stages: dict | None = None,
    source_run_manifest_path: Path | None = None,
) -> bool | None:
    """Validate a real online crawl against its own durable run and source manifests.

    The independent source run is not required to share the native training smoke's
    run ID. Fixture-based pipeline tests are not evidence of external retrieval.
    """
    try:
        # RC certification must be tied to a completed independent run, not just
        # counters in a source manifest. Without the run manifest, evidence is
        # unknown rather than a pass.
        if source_run_manifest_path is None:
            return None
        run_manifest = json.loads(source_run_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(run_manifest, dict):
            return False
        if run_manifest.get("final_status") != "PASS":
            return False
        if run_manifest.get("errors") or run_manifest.get("degraded_stages"):
            return False
        expected_run_id = run_manifest.get("run_id")
        stages = run_manifest.get("stages")
        if not isinstance(expected_run_id, str) or not expected_run_id:
            return False
        if not isinstance(stages, dict):
            return False
        crawl_stage = stages.get("crawl")
        if not isinstance(crawl_stage, dict) or crawl_stage.get("status") != "PASS":
            return False
        if crawl_stage.get("errors"):
            return False
        stage_start = datetime.fromisoformat(str(crawl_stage.get("start", "")).replace("Z", "+00:00"))
        stage_end = datetime.fromisoformat(str(crawl_stage.get("end", "")).replace("Z", "+00:00"))
        if stage_start.tzinfo is None or stage_end.tzinfo is None or stage_end < stage_start:
            return False

        data = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("schema", 0)) < 3:
            return False
        if data.get("run_id") != expected_run_id:
            return False
        if not _is_sha256(data.get("source_definition_sha256")):
            return False
        if not data.get("retrieval_started_at") or not data.get("retrieval_completed_at"):
            return False
        started = datetime.fromisoformat(str(data["retrieval_started_at"]).replace("Z", "+00:00"))
        completed = datetime.fromisoformat(str(data["retrieval_completed_at"]).replace("Z", "+00:00"))
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            return False
        if started < stage_start or completed > stage_end:
            return False

        sources = data.get("sources")
        if not isinstance(sources, list) or not sources:
            return False
        configured_kinds = {
            source.get("kind") for source in sources
            if isinstance(source, dict)
        } & {"web", "github", "arxiv", "huggingface", "google"}
        if not configured_kinds:
            return False
        retrieval = data.get("retrieval")
        if not isinstance(retrieval, list) or not retrieval:
            return False

        successful_counts: dict[str, int] = {}
        for record in retrieval:
            if not isinstance(record, dict) or record.get("kind") not in configured_kinds:
                continue
            if str(record.get("status", "")).upper() != "SUCCESS":
                continue
            stats = record.get("stats")
            if not isinstance(stats, dict):
                stats = record
            documents = stats.get("document_count", 0)
            requests = stats.get("request_count", 0)
            if not _is_nonnegative_int(documents) or not _is_nonnegative_int(requests):
                return False
            if documents > 0 and requests > 0:
                kind = str(record["kind"])
                successful_counts[kind] = successful_counts.get(kind, 0) + documents
        if not successful_counts:
            return False

        # Bind claimed retrieval to the actual crawl artifact. The run manifest
        # must contain a SHA-256-addressed crawl output that still matches disk;
        # that JSONL must contain non-empty HTTP(S) documents from the claimed
        # online source kinds. Positive counters alone are not retrieval proof.
        outputs = crawl_stage.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        verified_documents: dict[str, int] = {}
        for artifact in outputs:
            if not isinstance(artifact, dict):
                continue
            artifact_path = artifact.get("path")
            artifact_digest = artifact.get("sha256")
            if not isinstance(artifact_path, str) or not _is_sha256(artifact_digest):
                continue
            path = Path(artifact_path)
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            try:
                path.relative_to(ROOT.resolve())
            except ValueError:
                return False
            if not path.is_file() or _sha256(path) != artifact_digest:
                return False
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        document = json.loads(line)
                        if not isinstance(document, dict):
                            return False
                        kind = document.get("source")
                        url = document.get("url")
                        content = document.get("text")
                        if kind not in successful_counts or not isinstance(url, str):
                            continue
                        parsed_url = urlsplit(url)
                        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
                            continue
                        if not isinstance(content, str) or not content.strip():
                            continue
                        verified_documents[str(kind)] = verified_documents.get(str(kind), 0) + 1
            except (OSError, UnicodeError, ValueError, TypeError):
                return False
        return any(verified_documents.get(kind, 0) > 0 for kind in successful_counts)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, json.JSONDecodeError):
        return False


def _harvest_native_evidence(
    *,
    output: Path,
    run_id: str,
    run_manifest_path: Path,
) -> None:
    """Validate and retain durable evidence from the completed native smoke run."""
    evidence: dict[str, object] = {}

    try:
        if not run_manifest_path.is_file():
            raise RuntimeError(f"missing run manifest: {run_manifest_path}")

        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id:
            raise RuntimeError(
                f"run id mismatch: expected={run_id} actual={manifest.get('run_id')}"
            )
        if manifest.get("final_status") != "PASS":
            raise RuntimeError(
                f"run manifest final_status={manifest.get('final_status')!r}"
            )
        if manifest.get("errors"):
            raise RuntimeError(f"run manifest contains errors: {manifest['errors']}")
        if manifest.get("degraded_stages"):
            raise RuntimeError(
                f"run manifest contains degraded stages: {manifest['degraded_stages']}"
            )

        identities = manifest.get("configuration")
        if not isinstance(identities, dict) or not identities:
            raise RuntimeError("run manifest has no config identities")

        dataset_report = run_manifest_path.parent / "dataset_report.json"
        if not dataset_report.is_file():
            raise RuntimeError(f"missing dataset report: {dataset_report}")

        dataset_payload = json.loads(
            dataset_report.read_text(encoding="utf-8")
        )
        dataset_sha = _sha256(dataset_report)
        recorded_dataset = manifest.get("dataset_report") or {}
        if recorded_dataset.get("sha256") != dataset_sha:
            raise RuntimeError(
                "dataset report SHA-256 does not match run manifest"
            )

        stages = manifest.get("stages") or {}
        required_stages = ("tokenize", "shard", "train", "export")
        if any((stages.get(name) or {}).get("status") != "PASS"
               for name in required_stages):
            raise RuntimeError(
                "required native stages are not all PASS"
            )

        # The native release smoke deliberately uses a pre-weighted local fixture.
        # Online retrieval must be supplied as an independent completed crawl run;
        # never infer it from the offline integration smoke or synthetic smoke manifest.
        if SOURCE_MANIFEST_PATH is not None and SOURCE_RUN_MANIFEST_PATH is not None:
            evidence["source_retrieval"] = _live_source_retrieval_evidence_is_valid(
                source_manifest_path=SOURCE_MANIFEST_PATH,
                source_run_manifest_path=SOURCE_RUN_MANIFEST_PATH,
            )
        else:
            evidence["source_retrieval"] = None

        # Dataset evidence is intentionally bounded to the local native smoke fixture.
        evidence["dataset"] = _dataset_evidence_is_valid(
            dataset_payload,
            manifest.get("dataset_identity"),
            manifest.get("metrics"),
        )

        train_metrics = (
            manifest.get("metrics", {}).get("train")
            or stages.get("train", {}).get("metrics")
            or {}
        )

        preflight = train_metrics.get("preflight")
        if isinstance(preflight, dict):
            viable = preflight.get("viable")
            evidence["eta"] = viable is True
            evidence["hardware"] = bool(
                manifest.get("hardware")
                or preflight.get("hardware")
            )
        else:
            evidence["eta"] = None
            evidence["hardware"] = bool(manifest.get("hardware"))

        evidence["auto_sizing"] = _auto_sizing_evidence(train_metrics)

        evidence["evaluation"] = _evaluation_evidence_is_valid(train_metrics)

        evidence["reproducibility"] = _reproducibility_evidence_is_valid(train_metrics)

        # Checkpoint evidence.
        checkpoint_dir = output / "checkpoints"
        final_ckpt = checkpoint_dir / "ckpt_final.pt"
        best_ckpt = checkpoint_dir / "ckpt_best.pt"

        if not final_ckpt.is_file() or final_ckpt.stat().st_size <= 0:
            raise RuntimeError("final checkpoint missing or empty")
        if not best_ckpt.is_file() or best_ckpt.stat().st_size <= 0:
            raise RuntimeError("best checkpoint missing or empty")

        from pipeline.trainer.train import checkpoint_metadata

        final_meta = checkpoint_metadata(final_ckpt)
        best_meta = checkpoint_metadata(best_ckpt)

        if not isinstance(final_meta, dict) or not isinstance(best_meta, dict):
            raise RuntimeError("checkpoint metadata is invalid")

        _validate_final_checkpoint_metadata(final_meta)
        _validate_best_checkpoint_metadata(best_meta)
        evidence["checkpoint"] = True
        evidence["best_checkpoint"] = True

        # Provenance: checkpoint metadata must contain the immutable identity chain.
        provenance = (
            final_meta.get("provenance")
            or final_meta.get("training_metadata", {}).get("provenance")
        )

        if not isinstance(provenance, dict):
            raise RuntimeError("final checkpoint provenance is missing")

        checkpoint_identities = provenance.get("config_identities")
        if not isinstance(checkpoint_identities, dict):
            raise RuntimeError("checkpoint config identities are missing")

        _validate_checkpoint_identity_chain(identities, checkpoint_identities)

        evidence["provenance"] = True

        evidence["lineage"] = _lineage_evidence_is_valid(provenance, run_id)

        # SQLite must contain the same run and durable tracking evidence.
        db_path = output / "experiment.db"
        if not db_path.is_file():
            raise RuntimeError(f"missing experiment database: {db_path}")

        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT id, status, git_sha, git_branch FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()

            if row is None:
                raise RuntimeError(f"run {run_id} missing from experiment.db")

            if row["status"] != "PASS":
                raise RuntimeError(
                    f"SQLite run status is {row['status']!r}, expected PASS"
                )

            _validate_run_git_identity(
                manifest=manifest,
                database_git_sha=row["git_sha"],
                expected_git_sha=_git(ROOT, "rev-parse", "HEAD"),
            )

            stage_count = conn.execute(
                "SELECT COUNT(*) FROM stages WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

            checkpoint_count = conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

            hardware_count = conn.execute(
                "SELECT COUNT(*) FROM hardware WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

            evaluation_count = conn.execute(
                "SELECT COUNT(*) FROM evaluations WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]

            if stage_count < 4:
                raise RuntimeError("SQLite stage evidence is incomplete")
            if checkpoint_count < 2:
                raise RuntimeError("SQLite checkpoint evidence is incomplete")
            if hardware_count < 1:
                raise RuntimeError("SQLite hardware evidence is missing")
            if evaluation_count < 1:
                raise RuntimeError("SQLite evaluation evidence is missing")

        evidence["sqlite"] = True

        # Export/inference were already structurally validated by _native_smoke().
        evidence["export"] = True
        evidence["inference"] = True

    except Exception as exc:
        print(f"Native evidence validation failed: {exc}")
        validated_export = NATIVE_EVIDENCE.get("export") is True
        validated_inference = NATIVE_EVIDENCE.get("inference") is True
        validated_source_retrieval = NATIVE_EVIDENCE.get("source_retrieval") is True
        NATIVE_EVIDENCE.clear()
        NATIVE_EVIDENCE["source_retrieval"] = validated_source_retrieval
        NATIVE_EVIDENCE["provenance"] = False
        NATIVE_EVIDENCE["lineage"] = False
        NATIVE_EVIDENCE["dataset"] = False
        NATIVE_EVIDENCE["checkpoint"] = False
        NATIVE_EVIDENCE["best_checkpoint"] = False
        NATIVE_EVIDENCE["reproducibility"] = False
        NATIVE_EVIDENCE["hardware"] = False
        NATIVE_EVIDENCE["eta"] = False
        NATIVE_EVIDENCE["evaluation"] = False
        NATIVE_EVIDENCE["sqlite"] = False
        NATIVE_EVIDENCE["export"] = validated_export
        NATIVE_EVIDENCE["inference"] = validated_inference
        return

    # Preserve validated native evidence before TemporaryDirectory removes the run.
    import shutil
    evidence_root = ROOT / "release-evidence" / "native-smoke" / run_id
    evidence_root.mkdir(parents=True, exist_ok=True)
    durable_sources = (
        run_manifest_path,
        dataset_report,
        db_path,
        output.parent / "source_manifest.json",
    )
    for source in durable_sources:
        if source.is_file():
            shutil.copy2(source, evidence_root / source.name)
    for source in output.rglob("*"):
        if source.is_file() and (
            source.suffix.lower() == ".json" and "manifest" in source.name.lower()
        ):
            destination = evidence_root / "artifacts" / source.relative_to(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    NATIVE_EVIDENCE.update(evidence)


def _build_and_validate_wheel() -> tuple[bool, str]:
    """Build and validate a wheel in a disposable directory."""
    with tempfile.TemporaryDirectory(prefix="model-lab-wheel-") as tmp:
        wheel_dir = Path(tmp)
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "wheel", ".",
                "--no-deps", "--no-build-isolation",
                "--wheel-dir", str(wheel_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "wheel build failed").strip()
            return False, detail[-2000:]

        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            return False, f"Expected exactly one wheel, found {len(wheels)}"

        wheel = wheels[0]
        try:
            with zipfile.ZipFile(wheel) as archive:
                corrupt = archive.testzip()
                if corrupt is not None:
                    return False, f"Corrupt wheel member: {corrupt}"

                names = set(archive.namelist())
                suffixes = (
                    ".dist-info/METADATA",
                    ".dist-info/WHEEL",
                    ".dist-info/RECORD",
                    ".dist-info/entry_points.txt",
                )
                paths = {}
                for suffix in suffixes:
                    matches = [name for name in names if name.endswith(suffix)]
                    if len(matches) != 1:
                        return False, (
                            f"Wheel must contain exactly one {suffix}; "
                            f"found {len(matches)}"
                        )
                    paths[suffix] = matches[0]

                metadata = Parser().parsestr(
                    archive.read(paths[".dist-info/METADATA"]).decode("utf-8")
                )
                project = tomllib.loads(
                    (ROOT / "pyproject.toml").read_text(encoding="utf-8")
                )["project"]
                normalize = lambda value: value.lower().replace("_", "-")

                if normalize(metadata.get("Name", "")) != normalize(project["name"]):
                    return False, "Wheel package name does not match pyproject.toml"
                if metadata.get("Version") != project["version"]:
                    return False, "Wheel version does not match pyproject.toml"

                required_modules = {
                    "run_pipeline.py",
                    "mlframework.py",
                    "run_command_center.py",
                    "launch.py",
                    "scripts/__init__.py",
                    "scripts/export_gguf.py",
                    "scripts/export_cards.py",
                }
                missing = required_modules - names
                if missing:
                    return False, f"Missing wheel modules: {sorted(missing)}"

                required_config = {
                    "config/__init__.py",
                    "config/pipeline_config.yaml",
                    "config/pipeline_config.dataset.yaml",
                    "config/pipeline_config.full.yaml",
                    "config/pipeline_config.smoke.yaml",
                    "config/dataset_groups.yaml",
                    "config/dataset_groups.smoke.yaml",
                    "config/dataset_profiles.yaml",
                    "config/dataset_source_policy.yaml",
                    "config/source_weights.yaml",
                    "config/cleaner_config.yaml",
                    "config/seed_urls.txt",
                }
                missing_config = required_config - names
                if missing_config:
                    return False, (
                        f"Missing wheel configuration assets: {sorted(missing_config)}"
                    )

                parser = configparser.ConfigParser()
                parser.read_string(
                    archive.read(paths[".dist-info/entry_points.txt"]).decode("utf-8")
                )
                actual = (
                    dict(parser.items("console_scripts"))
                    if parser.has_section("console_scripts")
                    else {}
                )
                expected = {
                    "mlab": "run_pipeline:main",
                    "mlframework": "mlframework:main",
                    "mlab-command-center": "run_command_center:main",
                    "mlab-launch": "launch:main",
                }
                for name, target in expected.items():
                    if actual.get(name) != target:
                        return False, f"Missing or incorrect console entry point: {name}"

        except (
            OSError, ValueError, KeyError, zipfile.BadZipFile, configparser.Error
        ) as exc:
            return False, f"Wheel validation failed: {exc}"

        return True, f"Validated {wheel.name} ({wheel.stat().st_size} bytes)"


def _report_check(name: str, status: str, detail: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _write_release_report(
    *,
    static_failures: list[str],
    native_requested: bool,
    native_rc: int,
    profile: str | None = None,
) -> int:
    """Write a verification report without confusing smoke evidence with RC certification."""
    profile = profile or ("rc" if native_requested else "static")
    if profile not in {"static", "native-smoke", "rc"}:
        raise ValueError(f"Unknown release verification profile: {profile}")

    evidence = NATIVE_EVIDENCE
    checks: list[dict[str, str]] = []

    def add(name: str, ok: bool | None, detail: str = "") -> None:
        status = "PASS" if ok is True else "FAIL" if ok is False else "UNKNOWN"
        checks.append(_report_check(name, status, detail))

    add("tests", "pytest" not in static_failures, "compile/lint/pytest gate")
    add("configs", "doctor-required" not in static_failures, "canonical configuration and doctor")

    native_checks = (
        "provenance", "lineage", "dataset", "source_retrieval", "checkpoint",
        "best_checkpoint", "reproducibility", "hardware", "auto_sizing", "eta",
        "evaluation", "sqlite", "export", "inference",
    )
    for name in native_checks:
        value = evidence.get(name)
        if name == "auto_sizing" and isinstance(value, bool):
            # Both enabled and intentionally not needed are valid executed outcomes.
            add(
                name,
                True,
                "auto-sizing evaluated: enabled" if value
                else "auto-sizing evaluated: no resize needed",
            )
        else:
            add(
                name,
                bool(value) if value is not None else None,
                "native release evidence" if value is not None else "not executed",
            )

    add(
        "ui_backend_connectivity",
        bool(evidence["ui_backend_connectivity"])
        if "ui_backend_connectivity" in evidence else None,
        "localhost FastAPI probe"
        if "ui_backend_connectivity" in evidence else "run with --ui-probe",
    )
    add(
        "clean_clone",
        bool(evidence["clean_clone"]) if "clean_clone" in evidence else None,
        "temporary clone acceptance path"
        if "clean_clone" in evidence else "run with --clean-clone",
    )
    add(
        "documentation",
        (ROOT / "README.md").is_file()
        and (ROOT / "docs" / "development" / "START_HERE.md").is_file(),
        "release docs present",
    )
    secret_value = evidence.get("secrets")
    add(
        "secrets",
        bool(secret_value) if secret_value is not None else None,
        "security gate",
    )

    git_status = _git_state(ROOT)
    add(
        "git_state",
        git_status["commit"] is not None and not git_status["dirty"],
        json.dumps(git_status, sort_keys=True),
    )

    package_value = evidence.get("package")
    package_status = (
        package_value if package_value is True or package_value is False else None
    )
    add(
        "package",
        package_status,
        str(evidence.get("package_detail", "package build not executed")),
    )

    native_artifacts_ok = (
        evidence.get("export") is True and evidence.get("inference") is True
    )
    add(
        "release_artifacts",
        native_artifacts_ok if native_requested else None,
        "native GGUF export and inference evidence",
    )

    # Static verification, native smoke, and full RC certification have
    # deliberately different acceptance criteria.
    required_by_profile = {
        "static": {"tests", "configs", "documentation", "secrets"},
        "native-smoke": {
            "tests", "configs", "provenance", "lineage", "dataset",
            "checkpoint", "best_checkpoint", "reproducibility", "hardware",
            "eta", "evaluation", "sqlite", "export", "inference",
            "documentation", "secrets", "release_artifacts",
        },
        "rc": {check["name"] for check in checks},
    }
    required_names = set(required_by_profile[profile])
    if profile == "native-smoke":
        if "ui_backend_connectivity" in evidence:
            required_names.add("ui_backend_connectivity")
        if "clean_clone" in evidence:
            required_names.add("clean_clone")

    required = [check for check in checks if check["name"] in required_names]
    blockers = [
        f"{check['name']}: {check['status']}"
        for check in required
        if check["status"] != "PASS"
    ]

    gate_pass = (
        not static_failures
        and all(check["status"] == "PASS" for check in required)
        and (
            profile == "static"
            or (native_requested and native_rc == 0)
        )
    )

    report = {
        "schema": 1,
        "release": {
            "static": "ML-FRAMEWORK STATIC VERIFICATION",
            "native-smoke": "ML-FRAMEWORK NATIVE SMOKE",
            "rc": "ML-FRAMEWORK RC",
        }[profile],
        "profile": profile,
        "status": "PASS" if gate_pass else "FAIL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_status,
        "python": sys.version,
        "platform": platform.platform(),
        "checks": checks,
        "required_checks": sorted(required_names),
        "native_requested": native_requested,
        "native_return_code": native_rc,
        "failures": list(dict.fromkeys([*static_failures, *blockers])),
        "native_evidence": evidence,
    }

    out = ROOT / "release-evidence"
    out.mkdir(exist_ok=True)
    path = out / "release_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Release report: {path}")
    return 0 if gate_pass else 2


def _ui_probe() -> bool:
    try:
        from fastapi.testclient import TestClient
        from command_center.web import app
        with TestClient(app) as client:
            response = client.get("/api/system")
            return response.status_code == 200 and response.json().get("version") == _project_version()
    except Exception as exc:
        print(f"UI/backend probe failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Model Lab production verification gate."); parser.add_argument("--bootstrap-native", action="store_true", help="bootstrap pinned llama.cpp and validate native GGUF inference"); parser.add_argument("--skip-tests", action="store_true", help="not allowed for production verification"); parser.add_argument("--verify-ollama", action="store_true", help="also verify every exported GGUF through Ollama")
    parser.add_argument("--ui-probe", action="store_true", help="probe the localhost FastAPI backend")
    parser.add_argument("--clean-clone", action="store_true", help="exercise the documented clone/install/doctor/smoke path in a temporary clone")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="source_manifest.json from an independently completed online crawl run",
    )
    parser.add_argument(
        "--source-run-manifest",
        type=Path,
        help="run_manifest.json from that same online crawl run",
    )
    parser.add_argument(
        "--profile",
        choices=("static", "native-smoke", "rc"),
        default=None,
        help="verification profile; rc requires native, UI, clean-clone, and independent online-crawl evidence",
    )
    args = parser.parse_args()
    global SOURCE_MANIFEST_PATH, SOURCE_RUN_MANIFEST_PATH
    SOURCE_MANIFEST_PATH = args.source_manifest
    SOURCE_RUN_MANIFEST_PATH = args.source_run_manifest
    if (SOURCE_MANIFEST_PATH is None) != (SOURCE_RUN_MANIFEST_PATH is None):
        parser.error("--source-manifest and --source-run-manifest must be supplied together")
    profile = args.profile or (
        "native-smoke" if args.bootstrap_native else "static"
    )
    if profile in {"native-smoke", "rc"} and not args.bootstrap_native:
        parser.error(f"--profile {profile} requires --bootstrap-native")
    if profile == "static" and args.bootstrap_native:
        parser.error("--profile static cannot be combined with --bootstrap-native")
    if args.clean_clone and not args.bootstrap_native:
        parser.error("--clean-clone requires --bootstrap-native")
    if profile == "rc" and not args.ui_probe:
        parser.error("--profile rc requires --ui-probe")
    if profile == "rc" and not args.clean_clone:
        parser.error("--profile rc requires --clean-clone")
    if profile == "rc" and SOURCE_MANIFEST_PATH is None:
        parser.error("--profile rc requires --source-manifest and --source-run-manifest")
    if args.skip_tests: print("RELEASE VERIFICATION FAILED: --skip-tests is not allowed for a production verification."); return 2
    failures: list[str] = []; compile_targets = ["pipeline", "command_center", "scripts", "ui", "run_pipeline.py", "run_command_center.py", "launch.py"]
    if run([sys.executable, "-m", "compileall", "-q", *compile_targets]): failures.append("compileall")
    if run(["ruff", "check", "--select", "F", *LINT_TARGETS]): failures.append("ruff")
    evidence = ROOT / "release-evidence"; evidence.mkdir(exist_ok=True)
    if run([sys.executable, "-m", "pytest", "-q", "--cov-report=json:release-evidence/coverage.json"]): failures.append("pytest")
    if run([sys.executable, "run_pipeline.py", "--doctor"]): failures.append("doctor-required")
    if failures:
        _write_release_report(static_failures=failures, native_requested=args.bootstrap_native, native_rc=2, profile=profile)
        print("RELEASE VERIFICATION FAILED:", ", ".join(failures))
        print("Platform:", platform.platform())
        return 2
    if args.ui_probe:
        NATIVE_EVIDENCE["ui_backend_connectivity"] = _ui_probe()
        if not NATIVE_EVIDENCE["ui_backend_connectivity"]:
            failures.append("ui-backend")
    security_rc = run([sys.executable, "scripts/security_gate.py"])
    NATIVE_EVIDENCE["secrets"] = security_rc == 0
    if security_rc:
        failures.append("security")
    if not args.bootstrap_native:
        static_rc = _write_release_report(
            static_failures=failures,
            native_requested=False,
            native_rc=0,
            profile="static",
        )
        if static_rc:
            print("STATIC VERIFICATION FAILED: report contains a required non-PASS check.")
            return 2
        print("STATIC VERIFICATION PASSED: syntax, lint, coverage, tests, and required environment checks are green.")
        print("Native artifact verification was not run. A production release requires --bootstrap-native.")
        return 0
    if run([sys.executable, "scripts/reconcile_environment.py", "--project-root", str(ROOT), "--ensure-llamacpp"]):
        _write_release_report(static_failures=["llama.cpp-bootstrap"], native_requested=True, native_rc=2)
        print("RELEASE VERIFICATION FAILED: llama.cpp-bootstrap")
        print("Platform:", platform.platform())
        return 2
    print("native export prerequisites are green")
    native_rc = _native_smoke(verify_ollama=args.verify_ollama)
    if native_rc:
        _write_release_report(static_failures=["native export/inference"], native_requested=True, native_rc=native_rc)
        print("RELEASE VERIFICATION FAILED: native export/inference")
        print("Platform:", platform.platform())
        return 2
    if args.clean_clone:
        with tempfile.TemporaryDirectory(prefix="model-lab-clean-clone-") as clone_dir:
            clone = Path(clone_dir) / "ML-Framework"
            remote = _git(ROOT, "config", "--get", "remote.origin.url") or "https://github.com/psfr4590-afk/ML-Framework.git"
            expected_commit = _git(ROOT, "rev-parse", "HEAD")
            clone_rc = run(["git", "clone", "--depth", "1", remote, str(clone)])
            if clone_rc or not expected_commit:
                NATIVE_EVIDENCE["clean_clone"] = False
            else:
                fetch = subprocess.run(["git", "fetch", "--depth", "1", "origin", expected_commit], cwd=clone, check=False)
                checkout = subprocess.run(["git", "checkout", "--detach", expected_commit], cwd=clone, check=False) if fetch.returncode == 0 else None
                checked_out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, check=False, capture_output=True, text=True) if checkout and checkout.returncode == 0 else None
                exact_source = bool(checked_out and checked_out.returncode == 0 and checked_out.stdout.strip() == expected_commit)
                venv_dir = clone / ".release-venv"
                venv_python = venv_dir / "bin" / "python"
                if sys.platform == "win32":
                    venv_python = venv_dir / "Scripts" / "python.exe"
                create_venv = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=clone, check=False) if exact_source else None
                install_torch = subprocess.run([str(venv_python), "-m", "pip", "install", "-r", "requirements-torch.txt", "--index-url", "https://download.pytorch.org/whl/cpu"], cwd=clone, check=False) if create_venv and create_venv.returncode == 0 else None
                install = subprocess.run([str(venv_python), "-m", "pip", "install", str(clone)], cwd=clone, check=False) if install_torch and install_torch.returncode == 0 else None
                # The pipeline doctor treats CUDA availability as informational and
                # keeps native llama.cpp optional for this CPU-only clean-clone smoke.
                doctor = subprocess.run([str(venv_python), str(clone / "run_pipeline.py"), "--doctor"], cwd=clone, check=False) if install and install.returncode == 0 else None
                smoke = subprocess.run([str(venv_python), str(clone / "mlframework.py"), "smoke"], cwd=clone, check=False) if doctor and doctor.returncode == 0 else None
                NATIVE_EVIDENCE["clean_clone"] = bool(exact_source and create_venv and create_venv.returncode == 0 and install_torch and install_torch.returncode == 0 and install and install.returncode == 0 and doctor and doctor.returncode == 0 and smoke and smoke.returncode == 0)
        if not NATIVE_EVIDENCE.get("clean_clone"):
            failures.append("clean-clone")
    if (args.ui_probe
            and not NATIVE_EVIDENCE.get("ui_backend_connectivity")
            and "ui-backend" not in failures):
        failures.append("ui-backend")
    if profile == "rc":
        package_ok, package_detail = _build_and_validate_wheel()
        NATIVE_EVIDENCE["package"] = package_ok
        NATIVE_EVIDENCE["package_detail"] = package_detail

    rc = _write_release_report(
        static_failures=failures,
        native_requested=True,
        native_rc=native_rc,
        profile=profile,
    )
    if rc:
        print(f"RELEASE VERIFICATION FAILED: {profile} report contains a required non-PASS check.")
        return 2
    if profile == "rc":
        print("RC VERIFICATION PASSED: all required RC checks are green.")
    else:
        print("NATIVE SMOKE VERIFICATION PASSED: native-smoke report is green.")
    return 0


if __name__ == "__main__": raise SystemExit(main())
