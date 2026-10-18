import json
from pathlib import Path

import pytest
try:
    import torch
except ImportError:
    torch = None
import yaml

from pipeline.trainer.model import LlamaModel, ModelConfig
from pipeline.integrity import sha256_file
from pipeline.trainer.train import best_checkpoint, latest_checkpoint, load_checkpoint, save_checkpoint, select_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def test_checkpoint_retention_is_configurable_and_generation_knob_is_not_dead_config():
    trainer = (ROOT / "pipeline" / "trainer" / "train.py").read_text(encoding="utf-8")
    assert 't.get("keep_checkpoints", 3)' in trainer
    assert 'keep=keep_checkpoints' in trainer
    assert "generate_every_steps" not in trainer

    for name in ("pipeline_config.yaml", "pipeline_config.smoke.yaml", "pipeline_config.full.yaml"):
        cfg = _load(name)
        train = cfg["train"]
        assert int(train["keep_checkpoints"]) >= 1
        assert "generate_every_steps" not in train


def test_latest_checkpoint_uses_highest_numeric_training_step(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    for name in ("ckpt_0000010.pt", "ckpt_0000009.pt"):
        path = ckpt_dir / name
        path.write_bytes(b"checkpoint")
        manifest = {
            "kind": "checkpoint",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        (ckpt_dir / f"{name}.manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    for name in ("ckpt_best_0000099.pt", "ckpt_final_0000100.pt"):
        path = ckpt_dir / name
        path.write_bytes(b"terminal")
        manifest = {
            "kind": "checkpoint",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        (ckpt_dir / f"{name}.manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    assert latest_checkpoint(ckpt_dir).name == "ckpt_0000010.pt"


def test_latest_checkpoint_requires_integrity_manifest(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    path = ckpt_dir / "ckpt_0000011.pt"
    path.write_bytes(b"checkpoint")

    assert latest_checkpoint(ckpt_dir) is None


def test_load_checkpoint_restores_model_optimizer_and_step(tmp_path):
    if torch is None:
        pytest.skip('torch not installed')
    model_cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=16)
    model = LlamaModel(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    saved = save_checkpoint(
        model,
        optimizer,
        scaler,
        step=7,
        val_loss=1.25,
        cfg={"seed": 42},
        out_dir=tmp_path,
        provenance={"contract": "resume-test"},
    )

    restored = LlamaModel(model_cfg)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    step = load_checkpoint(
        saved,
        restored,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
        device=torch.device("cpu"),
        expected_provenance={"contract": "resume-test"},
    )

    assert step == 7
    for name, value in source.items():
        assert torch.equal(restored.state_dict()[name], value + 1.0)


def test_resume_contract_uses_numbered_checkpoint_not_terminal_artifact(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    numbered = ckpt_dir / "ckpt_0000002.pt"
    numbered.write_bytes(b"checkpoint")
    manifest = {
        "kind": "checkpoint",
        "size": numbered.stat().st_size,
        "sha256": sha256_file(numbered),
    }
    (ckpt_dir / f"{numbered.name}.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    final = ckpt_dir / "ckpt_final_0000002.pt"
    best = ckpt_dir / "ckpt_best_0000002.pt"
    for path in (final, best):
        path.write_bytes(b"terminal")
        manifest = {
            "kind": "checkpoint",
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        (ckpt_dir / f"{path.name}.manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    selected = latest_checkpoint(ckpt_dir)
    assert selected == numbered
    assert selected != final
    assert selected != best


def test_load_checkpoint_rejects_stale_provenance(tmp_path):
    if torch is None:
        pytest.skip('torch not installed')
    path = tmp_path / "ckpt_0000001.pt"
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Keep the fixture above the loader's truncation guard so this test reaches
    # provenance validation rather than failing integrity validation first.
    torch.save(
        {
            "step": 1,
            "model": model.state_dict(),
            "padding": torch.zeros(1024, dtype=torch.float32),
        },
        path,
    )
    from pipeline.integrity import sha256_file

    manifest = {
        "schema": 2,
        "kind": "checkpoint",
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "step": 1,
        "val_loss": 1.0,
        "provenance": {"train_config_sha256": "old"},
    }
    (tmp_path / "ckpt_0000001.pt.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    try:
        load_checkpoint(
            path,
            model,
            optimizer=optimizer,
            expected_provenance={"train_config_sha256": "new"},
        )
    except RuntimeError as exc:
        assert "provenance mismatch" in str(exc).lower()
    else:
        raise AssertionError("stale checkpoint provenance was accepted")


def _checkpoint_fixture(tmp_path):
    if torch is None:
        pytest.skip('torch not installed')
    model_cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4, n_kv_heads=4, d_ffn=32, seq_len=16)
    model = LlamaModel(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scaler


def test_best_checkpoint_tracks_authoritative_validation_metric_and_replaces_prior(tmp_path):
    model, optimizer, scaler = _checkpoint_fixture(tmp_path)
    first = save_checkpoint(
        model, optimizer, scaler, step=700, val_loss=3.42, cfg={"seed": 42},
        out_dir=tmp_path, tag="best", provenance={"contract": "best-test"},
        best_step=700, best_val_loss=3.42,
        training_metadata={"model_parameters": sum(p.numel() for p in model.parameters())},
    )
    second = save_checkpoint(
        model, optimizer, scaler, step=1000, val_loss=3.24, cfg={"seed": 42},
        out_dir=tmp_path, tag="best", provenance={"contract": "best-test"},
        best_step=1000, best_val_loss=3.24,
        training_metadata={"model_parameters": sum(p.numel() for p in model.parameters())},
    )

    assert first == second
    assert best_checkpoint(tmp_path) == second
    assert select_checkpoint(tmp_path, "best") == second
    meta = json.loads(second.with_name(second.name + ".manifest.json").read_text(encoding="utf-8"))
    assert meta["step"] == 1000
    assert meta["val_loss"] == 3.24
    assert meta["best_step"] == 1000
    assert meta["best_val_loss"] == 3.24


def test_final_checkpoint_records_final_step_separately_from_best(tmp_path):
    model, optimizer, scaler = _checkpoint_fixture(tmp_path)
    final = save_checkpoint(
        model, optimizer, scaler, step=1200, val_loss=3.30, cfg={"seed": 42},
        out_dir=tmp_path, tag="final", provenance={"contract": "final-test"},
        best_step=1000, best_val_loss=3.24, final_step=1200,
        training_metadata={"tokens_processed": 1200 * 16},
    )
    assert select_checkpoint(tmp_path, "final") == final
    meta = json.loads(final.with_name(final.name + ".manifest.json").read_text(encoding="utf-8"))
    assert meta["step"] == 1200
    assert meta["final_step"] == 1200
    assert meta["best_step"] == 1000
    assert meta["best_val_loss"] == 3.24


def test_checkpoint_metadata_records_reproducibility_fields(tmp_path):
    model, optimizer, scaler = _checkpoint_fixture(tmp_path)
    path = save_checkpoint(
        model, optimizer, scaler, step=7, val_loss=1.25, cfg={"seed": 42},
        out_dir=tmp_path, provenance={"contract": "metadata-test"},
        training_metadata={
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "architecture": "test",
            "sequence_length": 16,
            "batch_size": 2,
            "effective_batch_size": 8,
            "gradient_accumulation": 4,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "scheduler": "cosine",
            "precision": "float32",
            "seed": 42,
            "total_steps": 100,
            "tokens_processed": 112,
            "training_duration_seconds": 1.0,
            "evaluation_duration_seconds": 0.2,
            "checkpoint_duration_seconds": 0.1,
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["training_metadata"]
    for key in (
        "model_parameters", "architecture", "sequence_length", "batch_size",
        "effective_batch_size", "gradient_accumulation", "optimizer", "lr",
        "scheduler", "precision", "seed", "total_steps", "tokens_processed",
        "training_duration_seconds", "evaluation_duration_seconds",
        "checkpoint_duration_seconds", "random_state", "random_state_sha256",
    ):
        assert key in meta
    assert "rng_state" in payload


def test_resume_restores_optimizer_and_rng_state_for_deterministic_continuation(tmp_path):
    model, optimizer, scaler = _checkpoint_fixture(tmp_path)
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    # Create optimizer state before saving.
    loss = model(x.long(), y.long())[1]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    rng_before = torch.get_rng_state()
    expected_next = torch.rand(4)
    torch.set_rng_state(rng_before)
    path = save_checkpoint(
        model, optimizer, scaler, step=7, val_loss=1.25, cfg={"seed": 42},
        out_dir=tmp_path, provenance={"contract": "resume-test"},
    )

    restored_model, restored_optimizer, restored_scaler = _checkpoint_fixture(tmp_path)
    load_checkpoint(
        path, restored_model, restored_optimizer, restored_scaler,
        device=torch.device("cpu"), expected_provenance={"contract": "resume-test"},
    )
    restored_next = torch.rand(4)
    assert torch.equal(restored_next, expected_next)
    assert restored_optimizer.state_dict()["state"]
