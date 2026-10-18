from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
try:
    import torch
except ImportError:
    torch = None
import yaml

from pipeline.orchestrator import Pipeline
from pipeline.trainer.model import LlamaModel, ModelConfig
from pipeline.integrity import write_manifest
from scripts.export_gguf import _enforce_training_provenance, _load_checkpoint, _map_state
from command_center.store import DatasetStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_default_config_is_project_rooted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = Pipeline()
    assert pipeline._project_root == REPO_ROOT
    assert pipeline._cfg_path == REPO_ROOT / "config" / "pipeline_config.yaml"


def test_pipeline_no_resume_disables_artifact_and_checkpoint_resume(tmp_path):
    source = REPO_ROOT / "config" / "pipeline_config.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["pipeline"]["resume"] = True
    config["train"]["resume"] = True
    config["pipeline"]["output_dir"] = str(tmp_path / "output")
    config["pipeline"]["scratch_dir"] = str(tmp_path / "scratch")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    pipeline = Pipeline(str(config_path), resume=False)
    assert pipeline._resume is False
    assert pipeline.cfg["train"]["resume"] is False


def test_pipeline_all_honors_configured_stage_enablement(monkeypatch, tmp_path):
    pipeline = Pipeline()
    calls: list[str] = []
    pipeline.cfg.setdefault("tokenizer", {})["output_path"] = str(tmp_path / "tokenizer")

    def materialize(name: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        return path

    def tokenize(path):
        calls.append("tokenize")
        tokenizer_path = Path(pipeline.cfg["tokenizer"]["output_path"]) / "tokenizer.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_text("{}", encoding="utf-8")
        Path(str(tokenizer_path) + ".manifest.json").write_text("{}", encoding="utf-8")
        return object()

    monkeypatch.setattr(pipeline, "stage_crawl", lambda group=None: calls.append("crawl") or materialize("crawl"))
    monkeypatch.setattr(pipeline, "stage_clean", lambda path: calls.append("clean") or materialize("clean"))
    monkeypatch.setattr(pipeline, "stage_embed_dedup", lambda path: calls.append("dedup") or materialize("dedup"))
    monkeypatch.setattr(pipeline, "stage_weight", lambda path: calls.append("weight") or materialize("weight"))
    monkeypatch.setattr(pipeline, "stage_tokenize", tokenize)
    monkeypatch.setattr(pipeline, "stage_shard", lambda path, tokenizer: calls.append("shard") or materialize("shard"))
    monkeypatch.setattr(pipeline, "stage_train", lambda: calls.append("train") or materialize("checkpoint"))
    monkeypatch.setattr(pipeline, "stage_export", lambda: calls.append("export"))

    pipeline.run("all")
    assert calls == ["crawl", "clean", "dedup", "weight", "tokenize", "shard", "train"]


def test_model_param_count_respects_weight_tying():
    if torch is None:
        pytest.skip('torch not installed')
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=16)
    model = LlamaModel(cfg)
    assert model.param_count() == sum(p.numel() for p in model.parameters())


def test_export_mapping_produces_standard_llama_tensor_names():
    if torch is None:
        pytest.skip('torch not installed')
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=16)
    model = LlamaModel(cfg)
    mapped = _map_state(model.state_dict(), cfg)
    required = {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    assert required <= set(mapped)


def test_export_load_checkpoint_uses_canonical_integrity_gate(tmp_path):
    if torch is None:
        pytest.skip('torch not installed')
    from pipeline.trainer.train import save_checkpoint

    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=16)
    model = LlamaModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = save_checkpoint(model, optimizer, scaler, 1, 2.5, {"model_preset": "test"}, tmp_path)

    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="integrity verification failed"):
        _load_checkpoint(path)


def test_dataset_store_reports_pipeline_artifacts_under_output(tmp_path, monkeypatch):
    import command_center.store as module
    monkeypatch.setattr(module, "DATASETS", tmp_path / "datasets")
    store = DatasetStore()
    meta = store.create("Test", group_config={"id": "custom", "name": "Test", "sources": {}})
    root = store.path(meta["id"])
    out = root / "output"
    (out / "tokenizer").mkdir(parents=True)
    tokenizer = out / "tokenizer" / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    write_manifest(tokenizer, kind="tokenizer")
    state = store.refresh_pipeline_state(meta["id"])
    assert state["stages"]["tokenize"] == "complete"


def test_checkpoint_roundtrip_integrity(tmp_path):
    if torch is None:
        pytest.skip('torch not installed')
    from pipeline.trainer.train import save_checkpoint, load_checkpoint
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=16)
    model = LlamaModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = save_checkpoint(model, optimizer, scaler, 1, 2.5, {"model_preset": "test"}, tmp_path)
    assert path.exists()
    assert path.with_name(path.name + ".manifest.json").exists()
    reloaded = LlamaModel(cfg)
    opt2 = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    scaler2 = torch.amp.GradScaler("cuda", enabled=False)
    assert load_checkpoint(path, reloaded, opt2, scaler2, torch.device("cpu")) == 1


def test_checkpoint_resume_restores_exact_loader_cursor(tmp_path):
    if torch is None:
        pytest.skip('torch not installed')
    from pipeline.shardwriter.shard_writer import ShardDataLoader
    from pipeline.trainer.train import load_checkpoint, save_checkpoint

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    files = []
    for split, start in (("train", 0), ("val", 100)):
        path = shard_dir / f"shard_00000_{split}.bin"
        path.write_bytes(np.arange(start, start + 64, dtype=np.uint16).tobytes())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"name": path.name, "size": path.stat().st_size, "sha256": digest})
    (shard_dir / "shards.manifest.json").write_text(
        json.dumps({"version": 1, "dtype": "uint16", "sequence_length": 8, "vocab_size": 200, "files": files}),
        encoding="utf-8",
    )

    train = ShardDataLoader(shard_dir, "train", 8, seed=17)
    val = ShardDataLoader(shard_dir, "val", 8, seed=17)
    model_cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=8)
    model = LlamaModel(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    provenance = {"schema": 1, "pipeline_config_sha256": "a" * 64, "train_config_sha256": "b" * 64, "model_config_sha256": "c" * 64, "shard_manifest_sha256": "d" * 64, "seed": 17}

    train.next_batch(1)
    val.next_batch(1)
    path = save_checkpoint(model, optimizer, scaler, 3, 1.5, {"seed": 17}, tmp_path / "checkpoints", provenance=provenance, train_loader=train, val_loader=val)
    expected_x, expected_y = train.next_batch(1)

    resumed_model = LlamaModel(model_cfg)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)
    resumed_train = ShardDataLoader(shard_dir, "train", 8, seed=17)
    resumed_val = ShardDataLoader(shard_dir, "val", 8, seed=17)
    assert load_checkpoint(path, resumed_model, resumed_optimizer, resumed_scaler, torch.device("cpu"), expected_provenance=provenance, train_loader=resumed_train, val_loader=resumed_val) == 3
    actual_x, actual_y = resumed_train.next_batch(1)
    assert torch.equal(actual_x, expected_x)
    assert torch.equal(actual_y, expected_y)


def test_export_provenance_rejects_missing_or_mismatched_lineage(tmp_path):
    with pytest.raises(RuntimeError, match="legacy provenance schema"):
        _enforce_training_provenance(tmp_path / "output", {"provenance": {}})


def test_semantic_dedup_has_dependency_free_fallback(monkeypatch):
    import pipeline.embedder.semantic_dedup as module
    from pipeline.types import Document

    monkeypatch.setattr(module, "ST_AVAILABLE", False)
    deduper = module.SemanticDeduplicator({"mode": "fallback", "similarity_threshold": 0.8})
    docs = [
        Document(doc_id="a", text="alpha beta gamma delta", final_weight=1.0),
        Document(doc_id="b", text="alpha beta gamma delta", final_weight=0.5),
        Document(doc_id="c", text="completely unrelated content", final_weight=1.0),
    ]
    kept = deduper.run(docs)
    assert [d.doc_id for d in kept] == ["a", "c"]


def test_shard_resume_rejects_same_size_hash_tampering(tmp_path, monkeypatch):
    config = yaml.safe_load((REPO_ROOT / "config" / "pipeline_config.yaml").read_text(encoding="utf-8"))
    config["pipeline"]["output_dir"] = str(tmp_path / "output")
    config["pipeline"]["scratch_dir"] = str(tmp_path / "scratch")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pipeline = Pipeline(str(config_path), resume=True)

    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    shard_dir = pipeline._out / "shards"
    shard_dir.mkdir(parents=True)
    shard = shard_dir / "shard_00000_train.bin"
    original = b"A" * 32
    shard.write_bytes(original)
    provenance = pipeline._provenance(
        "shard",
        input_path,
        {
            "shard_config_sha256": pipeline._hash_value(pipeline.cfg["shard"]) if hasattr(pipeline, "_hash_value") else hashlib.sha256(json.dumps(pipeline.cfg["shard"], sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest(),
            "tokenizer_sha256": None,
            "tokenizer_vocab_size": 32,
        },
    )
    marker = shard_dir / "shards.manifest.json"
    marker.write_text(json.dumps({"schema": 4, "files": [{"name": shard.name, "size": len(original), "sha256": hashlib.sha256(original).hexdigest()}], "provenance": provenance}), encoding="utf-8")

    shard.write_bytes(b"B" * len(original))
    called = []

    class FakeWriter:
        def __init__(self, cfg, tokenizer):
            pass

        def write(self, corpus_path):
            called.append(corpus_path)
            shard.write_bytes(original)

    monkeypatch.setattr("pipeline.orchestrator._load_class", lambda module, name: FakeWriter if name == "ShardWriter" else None)
    tokenizer = type("Tokenizer", (), {"get_vocab_size": lambda self: 32})()
    result = pipeline.stage_shard(input_path, tokenizer)
    assert result == shard_dir
    assert called == [input_path]
    assert shard.read_bytes() == original


def test_checkpoint_safe_loader_fails_closed_without_weights_only_support(tmp_path, monkeypatch):
    from pipeline.trainer import train as train_module

    observed = {}

    def unsupported_loader(path, **kwargs):
        observed.update(kwargs)
        raise TypeError("unexpected keyword argument 'weights_only'")

    if train_module.torch is None:
        from types import SimpleNamespace
        monkeypatch.setattr(train_module, "torch", SimpleNamespace(load=unsupported_loader))
    else:
        monkeypatch.setattr(train_module.torch, "load", unsupported_loader)
    with pytest.raises(RuntimeError, match="Safe checkpoint loading requires"):
        train_module._safe_torch_load(tmp_path / "checkpoint.pt", map_location="cpu")

    assert observed["weights_only"] is True
    assert observed["map_location"] == "cpu"


def test_checkpoint_restores_optimizer_and_rng_state(tmp_path):
    if torch is None:
        pytest.skip('torch not installed')
    import random

    from pipeline.trainer.train import save_checkpoint, load_checkpoint

    cfg = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        d_ffn=32,
        seq_len=16,
    )
    model = LlamaModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Populate AdamW state before saving, rather than testing an empty optimizer.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # Make the checkpoint RNG state reproducible.
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    checkpoint = save_checkpoint(
        model, optimizer, scaler, 1, 2.5,
        {"model_preset": "test"}, tmp_path,
    )

    # Record the sequence expected immediately after the saved RNG state.
    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(4)

    # Disturb every RNG, then create fresh objects before loading.
    random.seed(9876)
    np.random.seed(9876)
    torch.manual_seed(9876)

    resumed_model = LlamaModel(cfg)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=1e-3
    )
    resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)

    assert load_checkpoint(
        checkpoint,
        resumed_model,
        resumed_optimizer,
        resumed_scaler,
        torch.device("cpu"),
    ) == 1

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)

    # Verify the populated optimizer state survived serialization.
    for original_parameter, resumed_parameter in zip(
        model.parameters(), resumed_model.parameters()
    ):
        original_state = optimizer.state[original_parameter]
        resumed_state = resumed_optimizer.state[resumed_parameter]

        assert original_state.keys() == resumed_state.keys()
        assert original_state, "Expected populated AdamW state"

        for key, original_value in original_state.items():
            resumed_value = resumed_state[key]
            if torch.is_tensor(original_value):
                assert torch.equal(original_value, resumed_value), key
            else:
                assert original_value == resumed_value, key
