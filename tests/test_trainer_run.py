from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip('torch')

from pipeline.trainer import train as train_module


class TinyModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, y):
        prediction = self.weight * (x.float() / 512.0)
        loss = (prediction - y.float() / 512.0).pow(2).mean()
        return None, loss


def _make_shards(root: Path, seq_len: int = 4) -> Path:
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True)

    source_manifest = root.parent / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "status": "complete",
                "files": ["synthetic-source.jsonl"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train = np.arange(256, dtype=np.uint16)
    val = np.arange(256, 512, dtype=np.uint16)

    files = []
    for name, data in (
        ("shard_00000_train.bin", train),
        ("shard_00000_val.bin", val),
    ):
        path = shard_dir / name
        data.tofile(path)
        files.append(
            {
                "name": name,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    (shard_dir / "shards.manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "dtype": "uint16",
                "sequence_length": seq_len,
                "vocab_size": 512,
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return shard_dir


def _tiny_cfg(tmp_path: Path, shard_dir: Path, **overrides):
    train = {
        "allow_cpu_training": True,
        "model_preset": "85M",
        "vocab_size": 512,
        "seq_len": 4,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "total_steps": 3,
        "warmup_steps": 0,
        "lr_max": 1e-3,
        "lr_min": 1e-4,
        "eval_every_steps": 1,
        "eval_batches": 1,
        "checkpoint_every_steps": 1,
        "keep_checkpoints": 2,
        "resume": False,
        "shard_dir": str(shard_dir),
        "seed": 42,
    }
    train.update(overrides)
    return {
        "pipeline": {"output_dir": str(tmp_path)},
        "shard": {"sequence_length": 4, "dtype": "uint16"},
        "preflight": {"enabled": False},
        "train": train,
    }


@pytest.fixture
def tiny_training(monkeypatch):
    monkeypatch.setattr(
        train_module,
        "LlamaModel",
        lambda cfg: TinyModel(cfg),
    )
    monkeypatch.setattr(
        train_module,
        "profile_hardware",
        lambda: type(
            "Hardware",
            (),
            {
                "to_dict": lambda self: {
                    "gpu_name": "test",
                    "gpu_memory_gb": 1.0,
                }
            },
        )(),
    )


def test_auto_sizing_respects_shard_sequence_length(
    tmp_path, tiny_training, monkeypatch
):
    shard_dir = _make_shards(tmp_path, seq_len=128)
    captured = {}

    class Profile:
        model_preset = "85M"
        seq_len = 128
        batch_size = 1
        grad_accum_steps = 1
        eval_batches = 1
        eval_every_steps = 1
        checkpoint_every_steps = 1
        recommended_steps = 1

        def to_dict(self):
            return {"name": "cpu-safe", "seq_len": self.seq_len}

    def fake_recommend_training_profile(hardware, **kwargs):
        captured.update(kwargs)
        assert kwargs["max_seq_len"] == 128
        return Profile()

    monkeypatch.setattr(
        train_module,
        "recommend_training_profile",
        fake_recommend_training_profile,
    )
    monkeypatch.setattr(
        train_module,
        "estimate_total_tokens",
        lambda _shard_dir: 256,
    )

    cfg = _tiny_cfg(
        tmp_path,
        shard_dir,
        auto_size=True,
        seq_len=128,
        total_steps=1,
    )
    cfg["shard"]["sequence_length"] = 128

    trainer = train_module.Trainer(cfg)
    trainer.run()

    assert captured["max_seq_len"] == 128
    assert trainer.t_cfg["seq_len"] == 128


def test_trainer_run_completes_and_writes_final_checkpoint(
    tmp_path, tiny_training
):
    shard_dir = _make_shards(tmp_path)

    trainer = train_module.Trainer(
        _tiny_cfg(tmp_path, shard_dir, total_steps=2)
    )

    model, step = trainer.run()

    assert isinstance(model, TinyModel)
    assert step == 2

    final = tmp_path / "checkpoints" / "ckpt_final.pt"
    manifest = final.with_name(final.name + ".manifest.json")

    assert final.is_file()
    assert manifest.is_file()

    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["checkpoint_kind"] == "final"
    assert metadata["final_step"] == 2


def test_trainer_run_records_evaluation_and_best_checkpoint(
    tmp_path, tiny_training
):
    shard_dir = _make_shards(tmp_path)

    trainer = train_module.Trainer(
        _tiny_cfg(
            tmp_path,
            shard_dir,
            total_steps=2,
            checkpoint_every_steps=99,
        )
    )

    trainer.run()

    metrics = (tmp_path / "logs" / "metrics.jsonl").read_text(
        encoding="utf-8"
    )
    records = [json.loads(line) for line in metrics.splitlines()]

    assert any("val_loss" in record for record in records)

    best = tmp_path / "checkpoints" / "ckpt_best.pt"
    assert best.is_file()

    best_meta = json.loads(
        best.with_name(best.name + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert best_meta["checkpoint_kind"] == "best"
    assert best_meta["best_step"] is not None


def test_final_evaluation_creates_best_checkpoint_when_no_periodic_eval_runs(
    tmp_path, tiny_training, monkeypatch
):
    shard_dir = _make_shards(tmp_path)
    trainer = train_module.Trainer(
        _tiny_cfg(
            tmp_path,
            shard_dir,
            total_steps=2,
            eval_every_steps=99,
            checkpoint_every_steps=99,
        )
    )
    monkeypatch.setattr(trainer, "_eval", lambda *args: 1.25)

    _, step = trainer.run()
    assert step == 2

    best = tmp_path / "checkpoints" / "ckpt_best.pt"
    final = tmp_path / "checkpoints" / "ckpt_final.pt"
    assert best.is_file()
    assert final.is_file()

    best_manifest = json.loads(
        best.with_name(best.name + ".manifest.json").read_text(encoding="utf-8")
    )
    final_manifest = json.loads(
        final.with_name(final.name + ".manifest.json").read_text(encoding="utf-8")
    )
    assert best_manifest["checkpoint_kind"] == "best"
    assert best_manifest["step"] == 2
    assert best_manifest["best_step"] == 2
    assert best_manifest["best_val_loss"] == pytest.approx(1.25)
    assert final_manifest["best_step"] == 2
    assert final_manifest["best_val_loss"] == pytest.approx(1.25)

    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "metrics.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    final_evaluations = [
        record for record in records
        if record.get("evaluation") == "final"
    ]
    assert len(final_evaluations) == 1
    assert final_evaluations[0]["step"] == 2
    assert final_evaluations[0]["val_loss"] == pytest.approx(1.25)


def test_final_evaluation_does_not_replace_a_better_best_checkpoint(
    tmp_path, tiny_training, monkeypatch
):
    shard_dir = _make_shards(tmp_path)
    trainer = train_module.Trainer(
        _tiny_cfg(
            tmp_path,
            shard_dir,
            total_steps=2,
            eval_every_steps=1,
            checkpoint_every_steps=99,
        )
    )
    losses = iter([1.0, 1.5, 2.0])
    monkeypatch.setattr(trainer, "_eval", lambda *args: next(losses))

    trainer.run()

    best = tmp_path / "checkpoints" / "ckpt_best.pt"
    final = tmp_path / "checkpoints" / "ckpt_final.pt"
    best_manifest = json.loads(
        best.with_name(best.name + ".manifest.json").read_text(encoding="utf-8")
    )
    final_manifest = json.loads(
        final.with_name(final.name + ".manifest.json").read_text(encoding="utf-8")
    )

    assert best_manifest["step"] == 1
    assert best_manifest["val_loss"] == pytest.approx(1.0)
    assert best_manifest["best_step"] == 1
    assert best_manifest["best_val_loss"] == pytest.approx(1.0)
    assert final_manifest["val_loss"] == pytest.approx(2.0)
    assert final_manifest["best_step"] == 1
    assert final_manifest["best_val_loss"] == pytest.approx(1.0)


def test_trainer_run_prunes_periodic_checkpoints(
    tmp_path, tiny_training
):
    shard_dir = _make_shards(tmp_path)

    trainer = train_module.Trainer(
        _tiny_cfg(
            tmp_path,
            shard_dir,
            total_steps=4,
            eval_every_steps=99,
            checkpoint_every_steps=1,
            keep_checkpoints=2,
        )
    )

    trainer.run()

    numbered = sorted(
        (tmp_path / "checkpoints").glob("ckpt_[0-9]*.pt")
    )

    assert len(numbered) == 2
    assert [int(path.stem.split("_")[1]) for path in numbered] == [3, 4]


def test_trainer_refuses_cpu_without_explicit_opt_in(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        train_module.torch.cuda,
        "is_available",
        lambda: False,
    )

    cfg = {
        "pipeline": {"output_dir": str(tmp_path)},
        "train": {"allow_cpu_training": False},
        "preflight": {"enabled": False},
    }

    with pytest.raises(
        RuntimeError,
        match="allow_cpu_training=false",
    ):
        train_module.Trainer(cfg).run()


def test_trainer_closes_metrics_when_training_fails(
    tmp_path, tiny_training, monkeypatch
):
    shard_dir = _make_shards(tmp_path)

    def fail_after_start(self, batch_size):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(
        train_module.ShardDataLoader,
        "next_batch",
        fail_after_start,
    )

    trainer = train_module.Trainer(
        _tiny_cfg(tmp_path, shard_dir, total_steps=1)
    )

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        trainer.run()

    metrics_path = tmp_path / "logs" / "metrics.jsonl"
    assert metrics_path.is_file()

    metrics_path.unlink()


def test_latest_checkpoint_falls_back_from_corrupt_newest_checkpoint(
    tmp_path,
):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    old = ckpt_dir / "ckpt_1.pt"
    new = ckpt_dir / "ckpt_2.pt"

    old.write_bytes(b"valid-old")
    new.write_bytes(b"corrupt-new")

    old_manifest = {
        "kind": "checkpoint",
        "size": old.stat().st_size,
        "sha256": train_module.sha256_file(old),
    }
    new_manifest = {
        "kind": "checkpoint",
        "size": new.stat().st_size,
        "sha256": "0" * 64,
    }

    old.with_name(old.name + ".manifest.json").write_text(
        json.dumps(old_manifest),
        encoding="utf-8",
    )
    new.with_name(new.name + ".manifest.json").write_text(
        json.dumps(new_manifest),
        encoding="utf-8",
    )

    assert train_module.latest_checkpoint(ckpt_dir) == old
def test_trainer_run_resumes_from_checkpoint(
    tmp_path, tiny_training
):
    shard_dir = _make_shards(tmp_path)

    first = train_module.Trainer(
        _tiny_cfg(
            tmp_path,
            shard_dir,
            total_steps=2,
            checkpoint_every_steps=1,
            eval_every_steps=99,
            resume=False,
        )
    )

    _, first_step = first.run()
    assert first_step == 2

    second = train_module.Trainer(
        _tiny_cfg(
            tmp_path,
            shard_dir,
            total_steps=4,
            checkpoint_every_steps=1,
            eval_every_steps=99,
            resume=True,
        )
    )

    _, resumed_step = second.run()

    assert resumed_step == 4

    final = tmp_path / "checkpoints" / "ckpt_final.pt"
    manifest = final.with_name(final.name + ".manifest.json")

    assert final.is_file()
    assert manifest.is_file()

    metadata = json.loads(
        manifest.read_text(encoding="utf-8")
    )

    assert metadata["final_step"] == 4

def test_resume_allows_continuation_controls_to_change(tmp_path, tiny_training):
    shard_dir = _make_shards(tmp_path)

    first_cfg = _tiny_cfg(
        tmp_path,
        shard_dir,
        total_steps=2,
        checkpoint_every_steps=1,
        eval_every_steps=99,
        resume=False,
    )
    first = train_module.Trainer(first_cfg)
    _, first_step = first.run()
    assert first_step == 2

    second_cfg = _tiny_cfg(
        tmp_path,
        shard_dir,
        total_steps=4,
        checkpoint_every_steps=2,
        eval_every_steps=2,
        resume=True,
    )
    second = train_module.Trainer(second_cfg)
    _, resumed_step = second.run()

    assert resumed_step == 4


def test_resume_rejects_incompatible_training_configuration(tmp_path, tiny_training):
    shard_dir = _make_shards(tmp_path)

    first_cfg = _tiny_cfg(
        tmp_path,
        shard_dir,
        total_steps=2,
        resume=False,
    )
    first = train_module.Trainer(first_cfg)
    _, first_step = first.run()
    assert first_step == 2

    incompatible_cfg = _tiny_cfg(
        tmp_path,
        shard_dir,
        total_steps=4,
        resume=True,
    )
    incompatible_cfg["train"]["lr_max"] = 1.0e-4

    second = train_module.Trainer(incompatible_cfg)

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        second.run()




def test_load_checkpoint_raises_on_truncated_file(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    path = ckpt_dir / "ckpt_0000001.pt"
    path.write_bytes(b"too small")

    with pytest.raises(RuntimeError, match="Checkpoint missing or truncated"):
        train_module.load_checkpoint(path, TinyModel({}))


def test_load_checkpoint_raises_on_missing_manifest(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    path = ckpt_dir / "ckpt_0000001.pt"
    path.write_bytes(b"x" * 2048)
    # no manifest written

    with pytest.raises(RuntimeError, match="Checkpoint integrity manifest missing"):
        train_module.load_checkpoint(path, TinyModel({}))


def test_load_checkpoint_raises_on_integrity_failure(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    path = ckpt_dir / "ckpt_0000001.pt"
    path.write_bytes(b"x" * 2048)

    manifest = path.with_name(path.name + ".manifest.json")
    manifest.write_text(
        json.dumps({
            "kind": "checkpoint",
            "size": path.stat().st_size,
            "sha256": "a" * 64,   # wrong hash
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Checkpoint integrity verification failed"):
        train_module.load_checkpoint(path, TinyModel({}))


def test_latest_checkpoint_returns_none_when_all_checkpoints_are_corrupt(
    tmp_path, tiny_training
):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    corrupt = ckpt_dir / "ckpt_0000001.pt"
    corrupt.write_bytes(b"x" * 2048)
    manifest = corrupt.with_name(corrupt.name + ".manifest.json")
    manifest.write_text(
        json.dumps({
            "kind": "checkpoint",
            "size": corrupt.stat().st_size,
            "sha256": "0" * 64,
        }),
        encoding="utf-8",
    )

    assert train_module.latest_checkpoint(ckpt_dir) is None
