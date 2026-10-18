"""Independent Phase 1 artifact-lineage validator.

This command is read-only. It validates cryptographic integrity and explicit
configuration relationships without rewriting historical artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def artifact_id(kind: str, digest: str) -> str:
    return f"{kind}:{digest}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def validate(output_dir: Path, checkpoint: Path | None = None) -> tuple[list[dict[str, str]], int]:
    root = output_dir.resolve()
    scratch = root.parent / "scratch"
    source_manifest = root.parent / "source_manifest.json"
    weighted_manifest = scratch / "04_weighted.jsonl.manifest.json"
    tokenizer_manifest = root / "tokenizer" / "tokenizer.json.manifest.json"
    tokenizer = root / "tokenizer" / "tokenizer.json"
    shard_manifest = root / "shards" / "shards.manifest.json"
    checks: list[dict[str, str]] = []

    paths = {
        "Source manifest": source_manifest,
        "Weighted manifest": weighted_manifest,
        "Weighted artifact": scratch / "04_weighted.jsonl",
        "Tokenizer manifest": tokenizer_manifest,
        "Tokenizer": tokenizer,
        "Shard manifest": shard_manifest,
    }
    for name, path in paths.items():
        checks.append(result(name, "PASS" if path.is_file() else "FAIL", str(path)))

    if any(c["status"] == "FAIL" for c in checks):
        return checks, 2

    source = load_json(source_manifest)
    weighted = load_json(weighted_manifest)
    tokenizer_meta = load_json(tokenizer_manifest)
    shard = load_json(shard_manifest)

    weighted_ok = weighted.get("sha256") == sha256_file(scratch / "04_weighted.jsonl") if (scratch / "04_weighted.jsonl").is_file() else False
    tokenizer_ok = tokenizer_meta.get("sha256") == sha256_file(tokenizer)
    shard_files_ok = True
    for item in shard.get("files", []):
        shard_path = root / "shards" / str(item.get("name", ""))
        if not shard_path.is_file() or item.get("sha256") != sha256_file(shard_path):
            shard_files_ok = False
            break
    checks.extend([
        result("Weighted artifact", "PASS" if weighted_ok else "FAIL", str(weighted.get("sha256", ""))),
        result("Tokenizer artifact", "PASS" if tokenizer_ok else "FAIL", str(tokenizer_meta.get("sha256", ""))),
        result("Shard artifacts", "PASS" if shard_files_ok else "FAIL", str(len(shard.get("files", []))) + " files"),
    ])

    checkpoint_meta = None
    if checkpoint is None:
        candidates = sorted((root / "checkpoints").glob("ckpt_*.pt.manifest.json"))
        checkpoint = Path(str(candidates[-1]).removesuffix(".manifest.json")) if candidates else None
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        manifest_path = Path(str(checkpoint) + ".manifest.json")
        if not checkpoint.is_file() or not manifest_path.is_file():
            checks.append(result("Checkpoint", "FAIL", f"missing checkpoint or manifest: {checkpoint}"))
            return checks, 2
        checkpoint_meta = load_json(manifest_path)
        actual = sha256_file(checkpoint)
        checks.append(result("Checkpoint", "PASS" if actual == checkpoint_meta.get("sha256") else "FAIL",
                             f"sha256={actual}"))

    if int(source.get("schema", -1)) not in {2, 3} or not source.get("run_id") or not source.get("source_definition_sha256"):
        checks.append(result("Configuration", "FAIL", "source manifest has an unsupported schema or lacks Phase 1 configuration identity"))
        return checks, 2

    source_defs = {
        name: value.get("sha256")
        for name, value in (source.get("source_definition_files") or {}).items()
        if isinstance(value, dict)
    }
    source_def_sha = stable_hash(source_defs)
    checks.append(result("Source configuration", "PASS" if source_def_sha == source.get("source_definition_sha256")
                         else "FAIL", source_def_sha))

    export_manifest = root / "gguf" / "export_manifest.json"
    if export_manifest.is_file() and checkpoint is not None:
        export_data = load_json(export_manifest)
        export_ok = export_data.get("checkpoint_sha256") == sha256_file(checkpoint)
        checks.append(result("Export → checkpoint", "PASS" if export_ok else "FAIL",
                             export_data.get("checkpoint_sha256", "")))

    if checkpoint_meta is None:
        checks.append(result("Checkpoint lineage", "WARN", "no checkpoint supplied or discovered"))
        return checks, 0

    cp = checkpoint_meta.get("provenance") or {}
    if int(cp.get("schema", -1)) < 3:
        checks.append(result("Checkpoint lineage", "LEGACY",
                             "checkpoint uses legacy pipeline-wide provenance; preserved, not rewritten"))
        return checks, 2

    identities = cp.get("config_identities") or {}
    snapshots = cp.get("config_snapshots") or {}
    snapshots_ok = all((root / str(value)).is_file() for value in snapshots.values()) if snapshots else False
    checks.append(result("Configuration snapshots", "PASS" if snapshots_ok else "FAIL",
                         f"{len(snapshots)} immutable snapshot references"))
    stage_pairs = (
        ("Dataset lineage", weighted, "dataset_config_sha256"),
        ("Tokenizer lineage", tokenizer_meta, "tokenizer_config_sha256"),
        ("Shard lineage", shard, "shard_config_sha256"),
    )
    for name, manifest, key in stage_pairs:
        stage_prov = manifest.get("provenance") or {}
        stage_ids = stage_prov.get("config_identities") or {}
        ok = bool(identities.get(key)) and stage_ids.get(key) == identities.get(key) and stage_prov.get("run_id") == cp.get("run_id")
        checks.append(result(name, "PASS" if ok else "FAIL",
                             f"{key}={stage_ids.get(key)} run_id={stage_prov.get('run_id')}"))

    source_ok = cp.get("source_manifest_sha256") == sha256_file(source_manifest) and source.get("run_id") == cp.get("run_id")
    shard_ok = cp.get("shard_manifest_sha256") == sha256_file(shard_manifest)
    tokenizer_ok = (shard.get("provenance") or {}).get("tokenizer_sha256") == sha256_file(tokenizer)
    checks.extend([
        result("Checkpoint → source", "PASS" if source_ok else "FAIL", cp.get("source_manifest_sha256", "")),
        result("Checkpoint → shards", "PASS" if shard_ok else "FAIL", cp.get("shard_manifest_sha256", "")),
        result("Shards → tokenizer", "PASS" if tokenizer_ok else "FAIL", sha256_file(tokenizer)),
    ])

    parent_ids = set(cp.get("parent_artifact_ids") or [])
    expected_parents = {
        artifact_id("shard-manifest", sha256_file(shard_manifest)),
        artifact_id("tokenizer-manifest", sha256_file(tokenizer_manifest)),
        artifact_id("dataset-manifest", sha256_file(weighted_manifest)),
        artifact_id("source-manifest", sha256_file(source_manifest)),
    }
    parent_ok = expected_parents <= parent_ids
    checks.append(result("Parents", "PASS" if parent_ok else "FAIL",
                         f"expected={sorted(expected_parents)}"))

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    checks.append(result("Overall", overall, "Phase 1 lineage validation"))
    return checks, 0 if overall == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ML-Framework artifact lineage without modifying artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    checks, code = validate(Path(args.output_dir), Path(args.checkpoint) if args.checkpoint else None)
    print("LINEAGE VALIDATION")
    print("==================")
    for check in checks:
        print(f'{check["name"]:<24} {check["status"]:<7} {check["detail"]}')
    return code


if __name__ == "__main__":
    raise SystemExit(main())
