#!/usr/bin/env python3
"""Production export path: native checkpoint -> Hugging Face -> GGUF -> Ollama."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from pipeline.integrity import select_export_checkpoint, sha256_file, validate_checkpoint
from pipeline.tokenizer.special_tokens import special_token_ids
from pipeline.trainer.model import LlamaModel, ModelConfig
from scripts.export_cards import write_export_cards

log = logging.getLogger("export")
QUANTS = {"F16", "Q4_K_M", "Q5_K_M", "Q8_0"}
EXPORTER_VERSION = "9.0.0"
ALL_QUANTS = ("F16", "Q4_K_M", "Q5_K_M", "Q8_0")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Select the export checkpoint (final, best, latest) with a verified manifest."""
    return select_export_checkpoint(checkpoint_dir)


def _load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], ModelConfig, dict[str, Any]]:
    payload = validate_checkpoint(path); cfg = ModelConfig.from_dict(payload["model_cfg"]); state = payload["model"]
    if not isinstance(state, dict): raise RuntimeError(f"Checkpoint model state is not a mapping: {path}")
    model = LlamaModel(cfg)
    try: missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc: raise RuntimeError(f"Checkpoint tensor shapes do not match its model config: {exc}") from exc
    missing = [x for x in missing if x not in {"rope_cos", "rope_sin"}]
    if missing or unexpected: raise RuntimeError(f"Checkpoint state is incompatible. Missing={missing}, unexpected={unexpected}")
    return state, cfg, payload


def _enforce_training_provenance(output_dir: Path, payload: dict[str, Any]) -> None:
    """Refuse export unless the checkpoint and retained upstream artifacts form a verified lineage."""
    provenance = payload.get("provenance") or {}
    schema = int(provenance.get("schema", -1))
    if schema < 3:
        raise RuntimeError(
            "Checkpoint uses legacy provenance schema; configuration lineage is historical and cannot "
            "be upgraded in place. Preserve the checkpoint as legacy evidence and retrain only when a "
            "new reproducible checkpoint is required."
        )

    required = (
        "schema", "run_id", "config_identities", "train_config_sha256",
        "model_config_sha256", "shard_manifest_sha256", "source_manifest_sha256", "seed", "parent_artifact_ids", "tokenizer_artifact_id", "dataset_artifact_id", "source_artifact_id",
    )
    missing = [key for key in required if key not in provenance or provenance[key] in (None, "")]
    if missing:
        raise RuntimeError(f"Checkpoint provenance is incomplete; refusing export: {missing}")

    identities = provenance["config_identities"]
    if not isinstance(identities, dict):
        raise RuntimeError("Checkpoint configuration identities are invalid; refusing export")
    for key in ("dataset_config_sha256", "tokenizer_config_sha256", "shard_config_sha256", "source_definition_sha256"):
        if not identities.get(key):
            raise RuntimeError(f"Checkpoint is missing {key}; refusing export")

    shard_manifest = output_dir / "shards" / "shards.manifest.json"
    tokenizer_manifest = output_dir / "tokenizer" / "tokenizer.json.manifest.json"
    weighted_manifest = output_dir.parent / "scratch" / "04_weighted.jsonl.manifest.json"
    source_manifest = output_dir.parent / "source_manifest.json"
    for path in (shard_manifest, tokenizer_manifest, weighted_manifest, source_manifest):
        if not path.is_file():
            raise RuntimeError(f"Required provenance artifact is missing; refusing export: {path}")

    if sha256_file(source_manifest) != provenance["source_manifest_sha256"]:
        raise RuntimeError("Checkpoint source-manifest identity does not match current source manifest; refusing export")
    if sha256_file(shard_manifest) != provenance["shard_manifest_sha256"]:
        raise RuntimeError("Checkpoint shard-manifest identity does not match current shards; refusing export")

    try:
        shard_data = json.loads(shard_manifest.read_text(encoding="utf-8"))
        tokenizer_data = json.loads(tokenizer_manifest.read_text(encoding="utf-8"))
        weighted_data = json.loads(weighted_manifest.read_text(encoding="utf-8"))
        source_data = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Required provenance manifest is invalid; refusing export") from exc

    if int(source_data.get("schema", -1)) < 2 or not source_data.get("retrieval_started_at") or not source_data.get("retrieval_completed_at") or not source_data.get("sources"):
        raise RuntimeError("Source manifest is incomplete; refusing export")

    source_definition_hashes = {
        name: value.get("sha256")
        for name, value in (source_data.get("source_definition_files") or {}).items()
        if isinstance(value, dict)
    }
    source_definition_sha = __import__("hashlib").sha256(
        json.dumps(source_definition_hashes, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    if source_definition_sha != identities["source_definition_sha256"]:
        raise RuntimeError("Source-definition configuration identity does not match the checkpoint lineage; refusing export")

    if source_data.get("run_id") != provenance["run_id"]:
        raise RuntimeError("Checkpoint → source manifest run identity mismatch; refusing export")

    for label, data, identity_key in (
        ("tokenizer", tokenizer_data, "tokenizer_config_sha256"),
        ("weighted", weighted_data, "dataset_config_sha256"),
        ("shards", shard_data, "shard_config_sha256"),
    ):
        stage_prov = data.get("provenance") or {}
        if stage_prov.get("run_id") != provenance["run_id"]:
            raise RuntimeError(f"{label} run identity does not match the checkpoint lineage; refusing export")
        stage_ids = stage_prov.get("config_identities") or {}
        if stage_ids.get(identity_key) != identities[identity_key]:
            raise RuntimeError(
                f"{label} configuration identity does not match the checkpoint lineage; refusing export"
            )

    tokenizer_sha = sha256_file(output_dir / "tokenizer" / "tokenizer.json")
    if (shard_data.get("provenance") or {}).get("tokenizer_sha256") != tokenizer_sha:
        raise RuntimeError("Shard → tokenizer artifact relationship is invalid; refusing export")

    expected_parents = {
        f"shard-manifest:{sha256_file(shard_manifest)}",
        f"tokenizer-manifest:{sha256_file(tokenizer_manifest)}",
        f"dataset-manifest:{sha256_file(weighted_manifest)}",
        f"source-manifest:{sha256_file(source_manifest)}",
    }
    actual_parents = set(provenance.get("parent_artifact_ids") or [])
    if not expected_parents <= actual_parents:
        raise RuntimeError("Checkpoint parent artifact lineage is incomplete; refusing export")
    if provenance.get("tokenizer_artifact_id") != f"tokenizer-manifest:{sha256_file(tokenizer_manifest)}":
        raise RuntimeError("Checkpoint → tokenizer artifact relationship is invalid; refusing export")
    if provenance.get("dataset_artifact_id") != f"dataset-manifest:{sha256_file(weighted_manifest)}":
        raise RuntimeError("Checkpoint → dataset artifact relationship is invalid; refusing export")
    if provenance.get("source_artifact_id") != f"source-manifest:{sha256_file(source_manifest)}":
        raise RuntimeError("Checkpoint → source artifact relationship is invalid; refusing export")
    if int(tokenizer_data.get("vocab_size", -1)) != int(payload["model_cfg"].get("vocab_size", -2)):
        raise RuntimeError("Tokenizer/model vocabulary provenance mismatch; refusing export")

def _hf_config(cfg: ModelConfig, model_name: str, special_ids: dict[str, int]) -> dict[str, Any]:
    return {"architectures":["LlamaForCausalLM"],"model_type":"llama","torch_dtype":"float32","vocab_size":cfg.vocab_size,"hidden_size":cfg.d_model,"intermediate_size":cfg.d_ffn,"num_hidden_layers":cfg.n_layers,"num_attention_heads":cfg.n_heads,"num_key_value_heads":cfg.n_kv_heads,"max_position_embeddings":cfg.seq_len,"rms_norm_eps":cfg.norm_eps,"rope_theta":cfg.rope_theta,"attention_bias":cfg.bias,"mlp_bias":cfg.bias,"tie_word_embeddings":True,"bos_token_id":int(special_ids["bos"]),"eos_token_id":int(special_ids["eos"]),"pad_token_id":int(special_ids["pad"]),"transformers_version":"4.0+","_name_or_path":model_name}


def _map_state(state: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}; table = {"norm1.weight":"input_layernorm.weight","norm2.weight":"post_attention_layernorm.weight","attn.q_proj.weight":"self_attn.q_proj.weight","attn.k_proj.weight":"self_attn.k_proj.weight","attn.v_proj.weight":"self_attn.v_proj.weight","attn.o_proj.weight":"self_attn.o_proj.weight","ffn.gate.weight":"mlp.gate_proj.weight","ffn.up.weight":"mlp.up_proj.weight","ffn.down.weight":"mlp.down_proj.weight"}
    for key, value in state.items():
        if key in {"rope_cos", "rope_sin"}: continue
        if key == "embed.weight": mapped = "model.embed_tokens.weight"
        elif key == "norm.weight": mapped = "model.norm.weight"
        elif key == "lm_head.weight": mapped = "lm_head.weight"
        elif key.startswith("blocks."):
            parts = key.split("."); tail = ".".join(parts[2:])
            try: mapped = f"model.layers.{parts[1]}.{table[tail]}"
            except KeyError as exc: raise RuntimeError(f"Unsupported model tensor: {key}") from exc
        else: raise RuntimeError(f"Unsupported model tensor: {key}")
        out[mapped] = value.detach().cpu().contiguous()
    if "model.embed_tokens.weight" not in out or "lm_head.weight" not in out: raise RuntimeError("Export requires both embedding and lm_head tensors")
    if out["model.embed_tokens.weight"].shape != out["lm_head.weight"].shape: raise RuntimeError("Embedding and lm_head shapes differ; refusing export")
    return out


def _write_hf_checkpoint(state: dict[str, torch.Tensor], cfg: ModelConfig, out: Path, model_name: str, special_ids: dict[str, int]) -> Path:
    out.mkdir(parents=True, exist_ok=True); torch.save(_map_state(state, cfg), out / "pytorch_model.bin"); _atomic_json(out / "config.json", _hf_config(cfg, model_name, special_ids)); return out


def _tokenizer_files(tokenizer_dir: Path, hf_dir: Path) -> None:
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.exists(): raise FileNotFoundError(f"Tokenizer file does not exist: {tokenizer_json}")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = tokenizer_dir / name
        if src.exists(): shutil.copy2(src, hf_dir / name)


def _llama_version(llamacpp_dir: Path) -> str:
    manifest = llamacpp_dir / "VENDOR_MANIFEST.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8")); return f"{data.get('tag','unknown')}@{data.get('commit','unknown')}"
        except (OSError, json.JSONDecodeError): pass
    try:
        proc = subprocess.run(["git","rev-parse","HEAD"], cwd=llamacpp_dir, text=True, capture_output=True, check=False)
        if proc.returncode == 0: return proc.stdout.strip()
    except OSError: pass
    return "unknown"


def _find_converter(llamacpp_dir: Path) -> Path:
    for candidate in (llamacpp_dir / "convert_hf_to_gguf.py", llamacpp_dir / "convert_hf_to_gguf_update.py"):
        if candidate.is_file(): return candidate
    raise FileNotFoundError(f"llama.cpp converter not found under {llamacpp_dir}. Provide a current checkout containing convert_hf_to_gguf.py.")


def _run(cmd: list[str], cwd: Path) -> None:
    log.info("Running: %s", " ".join(cmd)); proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.stdout: log.info(proc.stdout.rstrip())
    if proc.returncode != 0:
        if proc.stderr: log.error(proc.stderr.rstrip())
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _convert_to_f16(converter: Path, hf_dir: Path, gguf_dir: Path) -> Path:
    gguf_dir.mkdir(parents=True, exist_ok=True); out = gguf_dir / "model-f16.gguf"; _run([sys.executable, str(converter), str(hf_dir), "--outfile", str(out), "--outtype", "f16"], converter.parent)
    if not out.exists() or out.stat().st_size <= 0: raise RuntimeError("llama.cpp conversion completed without producing a GGUF file")
    return out


def _find_quantizer(llamacpp_dir: Path) -> Path:
    names = ["llama-quantize", "llama-quantize.exe"]
    for base in (llamacpp_dir / "build", llamacpp_dir / "bin", llamacpp_dir):
        if base.is_dir():
            for name in names:
                matches = list(base.rglob(name))
                if matches: return matches[0]
    raise FileNotFoundError("llama-quantize executable not found; build the pinned llama.cpp checkout before requesting a quantized GGUF")


def _quantize(f16: Path, quantizer: Path, quant: str, gguf_dir: Path) -> Path:
    out = gguf_dir / f"model-{quant.lower()}.gguf"; _run([str(quantizer), str(f16), str(out), quant], quantizer.parent)
    if not out.exists() or out.stat().st_size <= 0: raise RuntimeError("llama-quantize completed without producing the requested GGUF")
    return out


def _artifact_manifest(
    *,
    quant: str,
    artifact: Path,
    checkpoint: Path,
    payload: dict[str, Any],
    cfg: ModelConfig,
    export_config: dict[str, Any],
    lineage: dict[str, Any],
    converter_version: str,
    hf_dir: Path,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema": 4,
        "exporter_version": EXPORTER_VERSION,
        "timestamp": now,
        "quantization": quant,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "model_config": cfg.to_dict(),
        "export_config": export_config,
        "converter_version": converter_version,
        "hugging_face": {
            "path": str(hf_dir),
            "files": {
                p.name: {"path": str(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                for p in sorted(hf_dir.iterdir()) if p.is_file()
            },
        },
        "artifact": {
            "path": str(artifact),
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        },
        "lineage": lineage,
    }


def _write_modelfile(path: Path, gguf: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"FROM {gguf.name}\n\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER repeat_penalty 1.1\n",
        encoding="utf-8",
    )


def export_checkpoint(
    output_dir: str | Path,
    llamacpp_dir: str | Path,
    quant: str = "Q4_K_M",
    model_name: str = "pretrain-model",
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Export one or every supported GGUF quantization with durable lineage evidence."""
    output_dir = Path(output_dir).resolve()
    llamacpp_dir = Path(llamacpp_dir).resolve()
    requested = str(quant).upper()
    if requested == "ALL":
        requested_quants = list(ALL_QUANTS)
    elif requested in QUANTS:
        requested_quants = [requested]
    else:
        raise ValueError(f"Unsupported quantization '{quant}'. Choose from ALL, {sorted(QUANTS)}")

    ckpt = Path(checkpoint).resolve() if checkpoint else _find_best_checkpoint(output_dir / "checkpoints")
    state, cfg, payload = _load_checkpoint(ckpt)
    _enforce_training_provenance(output_dir, payload)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"Tokenizer is missing: {tokenizer_json}")
    try:
        tok_data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Tokenizer JSON is invalid: {tokenizer_json}") from exc
    vocab = tok_data.get("model", {}).get("vocab", {})
    if not isinstance(vocab, dict) or len(vocab) != cfg.vocab_size:
        raise RuntimeError(f"Tokenizer vocab ({len(vocab) if isinstance(vocab, dict) else 'invalid'}) does not match model vocab_size ({cfg.vocab_size})")

    hf_dir = output_dir / "export" / "hf"
    gguf_dir = output_dir / "gguf"
    special_ids = special_token_ids(tokenizer_dir)
    _write_hf_checkpoint(state, cfg, hf_dir, model_name, special_ids)
    _tokenizer_files(tokenizer_dir, hf_dir)
    converter = _find_converter(llamacpp_dir)
    converter_version = _llama_version(llamacpp_dir)
    f16 = _convert_to_f16(converter, hf_dir, gguf_dir)
    export_config = {
        "requested_quantization": requested,
        "quantizations": requested_quants,
        "model_name": model_name,
        "llamacpp_dir": str(llamacpp_dir),
        "converter": str(converter),
        "converter_version": converter_version,
        "hf_dir": str(hf_dir),
        "special_token_ids": special_ids,
    }
    provenance = payload.get("provenance") or {}
    lineage = {
        "run_id": provenance.get("run_id"),
        "parent_artifact_ids": list(provenance.get("parent_artifact_ids") or []),
        "training_provenance": provenance,
        "checkpoint_manifest": str(ckpt) + ".manifest.json",
        "hf_artifact": str(hf_dir),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for q in requested_quants:
        artifact = f16 if q == "F16" else _quantize(f16, _find_quantizer(llamacpp_dir), q, gguf_dir)
        modelfile = gguf_dir / f"Modelfile-{q}"
        _write_modelfile(modelfile, artifact)
        record = _artifact_manifest(
            quant=q,
            artifact=artifact,
            checkpoint=ckpt,
            payload=payload,
            cfg=cfg,
            export_config=export_config,
            lineage=lineage,
            converter_version=converter_version,
            hf_dir=hf_dir,
        )
        record["modelfile"] = str(modelfile)
        record["modelfile_sha256"] = sha256_file(modelfile)
        manifest_path = gguf_dir / f"export_manifest_{q}.json"
        _atomic_json(manifest_path, record)
        record["manifest"] = str(manifest_path)
        record["manifest_sha256"] = sha256_file(manifest_path)
        artifacts[q] = record

    primary = artifacts[requested_quants[-1]]
    aggregate = {
        "schema": 4,
        "exporter_version": EXPORTER_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "checkpoint_step": int(payload.get("step", 0)),
        "model_name": model_name,
        "model_config": cfg.to_dict(),
        "export_config": export_config,
        "converter_version": converter_version,
        "hf_dir": str(hf_dir),
        "hf_files": {p.name: sha256_file(p) for p in sorted(hf_dir.iterdir()) if p.is_file()},
        "artifacts": artifacts,
        "quantization": primary["quantization"],
        "f16_gguf": str(f16),
        "f16_gguf_sha256": sha256_file(f16),
        "final_gguf": primary["artifact"]["path"],
        "final_gguf_sha256": primary["artifact"]["sha256"],
        "final_gguf_size": primary["artifact"]["size_bytes"],
        "modelfile": primary["modelfile"],
        "lineage": lineage,
        "training_provenance": provenance,
    }
    card_paths = write_export_cards(output_dir, aggregate, payload)
    aggregate["dataset_card"] = card_paths["dataset_card"]
    aggregate["model_card"] = card_paths["model_card"]
    aggregate["dataset_card_sha256"] = sha256_file(Path(card_paths["dataset_card"]))
    aggregate["model_card_sha256"] = sha256_file(Path(card_paths["model_card"]))
    _atomic_json(gguf_dir / "export_manifest.json", aggregate)
    log.info("Export complete: %s", ", ".join(f"{q}={v['artifact']['path']}" for q, v in artifacts.items()))
    return aggregate

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Model Lab checkpoint to GGUF/Ollama"); parser.add_argument("--output-dir", required=True); parser.add_argument("--llamacpp-dir", required=True); parser.add_argument("--quant", default="Q4_K_M", choices=["ALL", *sorted(QUANTS)]); parser.add_argument("--model-name", default="pretrain-model"); parser.add_argument("--checkpoint"); parser.add_argument("--verbose", action="store_true"); args = parser.parse_args(argv); logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try: export_checkpoint(args.output_dir, args.llamacpp_dir, args.quant, args.model_name, args.checkpoint)
    except Exception as exc: log.error("Export failed: %s", exc); return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())