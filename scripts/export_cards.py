"""Generate auditable dataset and model cards for each model export."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    if not path.is_file(): return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file(): return None
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError): return None
    return value if isinstance(value, dict) else None


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip(): return result.stdout.strip()
    except OSError: pass
    return "unknown"


def _stage_manifest(scratch: Path, prefix: str) -> tuple[Path | None, dict[str, Any] | None]:
    matches = sorted(scratch.glob(f"{prefix}*.jsonl.manifest.json"))
    if not matches: return None, None
    path = matches[0]
    return path, _read_json(path)


def _shard_stats(shard_dir: Path) -> tuple[int, int | None, str | None]:
    manifest_path = shard_dir / "shards.manifest.json"; manifest = _read_json(manifest_path)
    if not manifest: return 0, None, None
    files = manifest.get("files", [])
    if not isinstance(files, list): return 0, None, _sha256(manifest_path)
    item_size = 4 if str(manifest.get("dtype", "uint16")) == "uint32" else 2
    total_bytes = sum(int(item.get("size", 0)) for item in files if isinstance(item, dict))
    return len(files), total_bytes // item_size, _sha256(manifest_path)


def _format_value(value: Any) -> str:
    if value is None: return "unknown"
    if isinstance(value, (dict, list)): return json.dumps(value, sort_keys=True)
    return str(value)


def write_export_cards(output_dir: str | Path, export_manifest: dict[str, Any], checkpoint_payload: dict[str, Any]) -> dict[str, str]:
    """Write auditable cards and refuse export-card generation without source provenance."""
    output_dir = Path(output_dir).resolve(); gguf_dir = output_dir / "gguf"; scratch = output_dir.parent / "scratch"; gguf_dir.mkdir(parents=True, exist_ok=True)
    stages = {"crawl": _stage_manifest(scratch, "01_crawled"), "clean": _stage_manifest(scratch, "02_cleaned"), "dedup": _stage_manifest(scratch, "03_deduped"), "weight": _stage_manifest(scratch, "04_weighted")}
    checkpoint_provenance = checkpoint_payload.get("provenance") or {}; train_cfg = checkpoint_payload.get("train_cfg") or {}; model_cfg = checkpoint_payload.get("model_cfg") or export_manifest.get("model_config") or {}
    shard_count, shard_tokens, shard_sha = _shard_stats(output_dir / "shards"); tokenizer_sha = _sha256(output_dir / "tokenizer" / "tokenizer.json")
    crawl_path, _ = stages["crawl"]; config_ids = checkpoint_provenance.get('config_identities') or {}
    if not config_ids and checkpoint_provenance.get('pipeline_config_sha256'):
        config_ids = {"pipeline_config_sha256": checkpoint_provenance.get('pipeline_config_sha256')}
    if not config_ids:
        for _, (_, manifest) in stages.items():
            if manifest:
                stage_prov = manifest.get("provenance") or {}
                config_ids = stage_prov.get("config_identities") or {}
                if not config_ids and stage_prov.get("pipeline_config_sha256"):
                    config_ids = {"pipeline_config_sha256": stage_prov.get("pipeline_config_sha256")}
                if config_ids:
                    break
    source_manifest = output_dir.parent / "source_manifest.json"
    if not source_manifest.is_file():
        raise RuntimeError(f"Source manifest is mandatory for production export provenance: {source_manifest}")
    source_manifest_sha = _sha256(source_manifest); git_commit = _git_commit(); dataset_name = train_cfg.get("dataset_name") or train_cfg.get("dataset_id") or output_dir.parent.name
    model_name = export_manifest.get("model_name", "pretrain-model"); final_gguf = Path(export_manifest.get("final_gguf", "")); final_hash = export_manifest.get("final_gguf_sha256") or _sha256(final_gguf); f16_hash = export_manifest.get("f16_gguf_sha256"); seed = train_cfg.get("seed", checkpoint_provenance.get("seed", "unknown"))
    dataset_lines = ["# Dataset Card", "", f"- Dataset/run: `{_format_value(dataset_name)}`", "- Description: Automatically generated at model export from retained pipeline artifacts.", "- Intended use: Training and evaluation of the exported local model.", "- Out-of-scope use: Treating this card as legal clearance, safety certification, or a guarantee of dataset quality.", "", "## Processing lineage", ""]
    for name, (_, manifest) in stages.items(): dataset_lines.append(f"- {name}: `{manifest.get('rows', 'unknown') if manifest else 'unknown'}` records")
    dataset_lines += [f"- Shards: `{shard_count}` files", f"- Sharded tokens: `{_format_value(shard_tokens)}`", f"- Tokenizer SHA-256: `{_format_value(tokenizer_sha)}`", "", "## Provenance", "", f"- Pipeline configuration SHA-256 (legacy compatibility evidence): `{_format_value(config_ids.get('pipeline_config_sha256'))}`", f"- Dataset configuration SHA-256: `{_format_value(config_ids.get('dataset_config_sha256'))}`", f"- Tokenizer configuration SHA-256: `{_format_value(config_ids.get('tokenizer_config_sha256'))}`", f"- Shard configuration SHA-256: `{_format_value(config_ids.get('shard_config_sha256'))}`", f"- Training configuration SHA-256: `{_format_value(config_ids.get('train_config_sha256') or checkpoint_provenance.get('train_config_sha256'))}`", f"- Model configuration SHA-256: `{_format_value(config_ids.get('model_config_sha256') or checkpoint_provenance.get('model_config_sha256'))}`", f"- Source-definition configuration SHA-256: `{_format_value(config_ids.get('source_definition_sha256'))}`", f"- Git commit: `{git_commit}`", f"- Crawl artifact manifest: `{crawl_path}`", f"- Crawl artifact manifest SHA-256: `{_sha256(crawl_path) if crawl_path else None}`", f"- Source manifest: `{source_manifest}`", f"- Source manifest SHA-256: `{_format_value(source_manifest_sha)}`", f"- Shard manifest SHA-256: `{_format_value(shard_sha)}`", "", "## Sources, licensing, and rights", "", "Review every source's URL or identifier, retrieval time, revision, license or usage terms, attribution requirements, restrictions, and raw-source SHA-256 before distribution. Public availability does not itself establish training rights.", "", "## Quality and limitations", "", "Record language/domain coverage, filtering and rejection rates, known extraction failures, duplication risk, and any dataset-specific quality evaluation here. Unknown values are intentionally retained as unknown rather than invented.", "", "## Removal and maintenance", "", "Rebuild affected downstream artifacts after source removal or takedown. Do not edit generated training data in place without regenerating its provenance chain.", ""]
    dataset_card = "\n".join(dataset_lines)
    model_lines = ["# Model Card", "", f"- Model name: `{model_name}`", "- Export format: GGUF", f"- Quantization: `{_format_value(export_manifest.get('quantization'))}`", f"- Checkpoint step: `{_format_value(export_manifest.get('checkpoint_step'))}`", f"- Model configuration: `{_format_value(model_cfg)}`", "", "## Intended use", "", "Local experimentation, evaluation, and inference within the operator's validated deployment context.", "", "## Training provenance", "", "- Dataset card: `DATASET_CARD.md`", f"- Training provenance: `{_format_value(checkpoint_provenance)}`", f"- Pipeline configuration SHA-256 (legacy compatibility evidence): `{_format_value(config_ids.get('pipeline_config_sha256'))}`", f"- Dataset configuration SHA-256: `{_format_value(config_ids.get('dataset_config_sha256'))}`", f"- Tokenizer configuration SHA-256: `{_format_value(config_ids.get('tokenizer_config_sha256'))}`", f"- Shard configuration SHA-256: `{_format_value(config_ids.get('shard_config_sha256'))}`", f"- Training configuration SHA-256: `{_format_value(config_ids.get('train_config_sha256') or checkpoint_provenance.get('train_config_sha256'))}`", f"- Model configuration SHA-256: `{_format_value(config_ids.get('model_config_sha256') or checkpoint_provenance.get('model_config_sha256'))}`", f"- Source-definition configuration SHA-256: `{_format_value(config_ids.get('source_definition_sha256'))}`", f"- Shard manifest SHA-256: `{_format_value(shard_sha)}`", f"- Seed: `{_format_value(seed)}`", f"- Host platform at export: `{platform.platform()}`", "", "## Evaluation", "", "No quality claim is inferred from export success. Record evaluation datasets, metrics, methodology, and known limitations here.", "", "## Export integrity", "", f"- Final GGUF SHA-256: `{_format_value(final_hash)}`", f"- F16 GGUF SHA-256: `{_format_value(f16_hash)}`", f"- llama.cpp converter revision: `{_format_value(export_manifest.get('converter_version'))}`", "- Export manifest: `export_manifest.json`", "", "## Limitations and risks", "", "The model may reproduce errors, bias, unsafe instructions, or other defects present in its training data. Export verification establishes artifact integrity, not model safety or usefulness.", "", "## License", "", "Confirm compatibility among the model license, training-data licenses, attribution requirements, and intended distribution before publishing.", ""]
    dataset_path = gguf_dir / "DATASET_CARD.md"; model_path = gguf_dir / "MODEL_CARD.md"; dataset_path.write_text(dataset_card, encoding="utf-8"); model_path.write_text("\n".join(model_lines), encoding="utf-8")
    return {"dataset_card": str(dataset_path), "model_card": str(model_path)}
