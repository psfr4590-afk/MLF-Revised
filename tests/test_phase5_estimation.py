from __future__ import annotations

import json

from pipeline.model_sizer import HardwareProfile, recommend_training_profile
from pipeline.trainer.model import ModelConfig
from pipeline.preflight import estimate_duration


def test_sub_4gb_profile_matches_low_vram_contract():
    hw = HardwareProfile("test", 4, 2.0, 1, 3.99, "test-gpu", True)
    profile = recommend_training_profile(hw)
    assert profile.model_preset == "85M"
    assert profile.seq_len == 128
    assert profile.batch_size == 1
    assert profile.grad_accum_steps == 32
    assert "VRAM < 4 GB" in profile.decision_reasons


def test_hardware_snapshot_contains_phase5_fields():
    hw = HardwareProfile(
        "test", 4, 2.0, 1, 3.99, "test-gpu", True,
        total_ram_gb=8.0, cpu_threads=8, cpu_name="test-cpu",
        cuda_version="12.8", driver_version="555.1", pytorch_version="2.14.0",
        os_name="Windows", disk_free_gb=50.0, disk_total_gb=100.0,
    )
    data = hw.to_dict()
    assert data["total_ram_gb"] == 8.0
    assert data["cpu_threads"] == 8
    assert data["driver_version"] == "555.1"
    assert data["pytorch_version"] == "2.14.0"
    assert data["disk_free_gb"] == 50.0


def test_duration_estimate_records_live_schedule():
    estimate = estimate_duration(
        total_steps=100, batch_size=1, grad_accum_steps=32, seq_len=128,
        tokens_per_sec=1280.0, eval_every_steps=20, eval_batches=2,
        checkpoint_every_steps=50,
    )
    assert estimate["estimated_duration_seconds"] is not None
    assert len([item for item in estimate["estimated_checkpoint_times"] if item.get("kind") == "scheduled"]) == 2
    assert len([item for item in estimate["estimated_eval_times"] if item.get("kind") != "final"]) == 5
    assert estimate["estimated_completion"] is not None


def test_duration_estimate_requires_benchmark_throughput():
    estimate = estimate_duration(
        total_steps=100, batch_size=1, grad_accum_steps=32, seq_len=128,
        tokens_per_sec=None, eval_every_steps=20, eval_batches=2,
        checkpoint_every_steps=50,
    )
    assert estimate["estimated_duration_seconds"] is None


def test_preflight_runtime_failure_is_non_viable(monkeypatch, tmp_path):
    from pipeline import preflight

    def fail_model(_train_cfg):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(preflight, "_make_model", fail_model)
    result = preflight.run_preflight({"model_preset": "85M", "seq_len": 128}, tmp_path)
    assert result.viable is False
    assert result.tokens_per_sec is None
    assert "out of memory" in result.failure.lower()


def test_preflight_report_persists_benchmark_and_estimate(tmp_path):
    from pipeline.preflight import PreflightResult, write_preflight_report

    result = PreflightResult(
        viable=True,
        tokens_per_sec=2048.0,
        steps_per_sec=16.0,
        peak_memory_gb=3.2,
        gpu_utilization_percent=87.0,
        benchmark_seconds=1.25,
        warmup_steps=1,
        benchmark_steps=3,
    )
    estimate = estimate_duration(
        total_steps=100,
        batch_size=1,
        grad_accum_steps=32,
        seq_len=128,
        tokens_per_sec=result.tokens_per_sec,
        eval_every_steps=20,
        eval_batches=2,
        checkpoint_every_steps=50,
    )
    path = tmp_path / "preflight_report.json"
    write_preflight_report(path, {
        "hardware": {"gpu_name": "test-gpu", "gpu_memory_gb": 3.99},
        "requested_configuration": {"model_preset": "117M", "seq_len": 512},
        "effective_configuration": {"model_preset": "85M", "seq_len": 128},
        "decision_reasons": ["VRAM < 4 GB"],
        "benchmark": result.to_dict(),
        "estimate": estimate,
    })
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["benchmark"]["tokens_per_sec"] == 2048.0
    assert data["benchmark"]["peak_memory_gb"] == 3.2
    assert data["estimate"]["estimated_duration_seconds"] is not None
    assert data["requested_configuration"]["model_preset"] == "117M"
    assert data["effective_configuration"]["model_preset"] == "85M"


def test_trainer_blocks_training_when_preflight_fails(monkeypatch, tmp_path):
    import pytest
    from pipeline.trainer import train as train_module
    from pipeline.model_sizer import HardwareProfile
    from pipeline.preflight import PreflightResult

    hardware = HardwareProfile("test", 4, 2.0, 1, 3.99, "test-gpu", True)

    monkeypatch.setattr(train_module, "profile_hardware", lambda: hardware)
    monkeypatch.setattr(
        train_module,
        "run_preflight",
        lambda *args, **kwargs: PreflightResult(
            viable=False,
            tokens_per_sec=None,
            steps_per_sec=None,
            peak_memory_gb=3.99,
            gpu_utilization_percent=None,
            benchmark_seconds=0.1,
            warmup_steps=1,
            benchmark_steps=3,
            failure="RuntimeError: out of memory",
        ),
    )

    def should_not_start(*args, **kwargs):
        raise AssertionError("model construction must not occur after a failed preflight")

    monkeypatch.setattr(train_module, "LlamaModel", should_not_start)
    cfg = {
        "pipeline": {"output_dir": str(tmp_path)},
        "train": {
            "allow_cpu_training": True,
            "model_preset": "85M",
            "seq_len": 128,
            "batch_size": 1,
            "grad_accum_steps": 1,
            "total_steps": 2,
        },
        "shard": {"sequence_length": 128},
        "preflight": {"enabled": True, "warmup_steps": 1, "benchmark_steps": 3},
    }

    with pytest.raises(RuntimeError, match="Preflight rejected training configuration"):
        train_module.Trainer(cfg).run()


def test_training_metadata_preserves_phase5_lineage(tmp_path):
    from pipeline.trainer.train import Trainer

    cfg = {
        "pipeline": {"output_dir": str(tmp_path)},
        "train": {},
    }
    trainer = Trainer(cfg)
    trainer._hardware_profile = {"gpu_name": "test-gpu", "gpu_memory_gb": 3.99}
    trainer._requested_configuration = {"model_preset": "117M", "seq_len": 512}
    trainer._initial_estimate_seconds = 123.5

    class FakeModel:
        def parameters(self):
            return []

    metadata = trainer._training_metadata(
        FakeModel(),
        {
            "model_preset": "85M",
            "seq_len": 128,
            "_hardware_profile": trainer._hardware_profile,
            "_preflight": {"viable": True, "tokens_per_sec": 2048.0},
        },
        seed=42,
        step=10,
        total_steps=100,
        batch_size=1,
        grad_accum=32,
        seq_len=128,
        tokens_processed=40960,
        start_time=__import__("time").time() - 10,
        evaluation_duration=1.0,
        checkpoint_duration=2.0,
    )

    assert metadata["hardware_profile"]["gpu_memory_gb"] == 3.99
    assert metadata["requested_configuration"]["model_preset"] == "117M"
    assert metadata["preflight"]["tokens_per_sec"] == 2048.0
    assert metadata["initial_estimate_seconds"] == 123.5
    assert metadata["estimated_vs_actual_seconds"]["estimated_seconds"] == 123.5
    assert metadata["estimated_vs_actual_seconds"]["actual_seconds"] >= 0


def test_checkpoint_manifest_preserves_phase5_metadata(tmp_path):
    import pytest
    torch = pytest.importorskip('torch')
    from pipeline.trainer.train import save_checkpoint

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.cfg = type("Cfg", (), {"to_dict": lambda self: {"test": True}})()

    class TinyScaler:
        def state_dict(self):
            return {"scale": 1.0}

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    provenance = {"schema": 3, "run_id": "test-run", "source_manifest_sha256": "a" * 64}
    training_metadata = {
        "hardware_profile": {"gpu_name": "test-gpu", "gpu_memory_gb": 3.99},
        "requested_configuration": {"model_preset": "117M"},
        "effective_configuration": {"model_preset": "85M"},
        "preflight": {"viable": True, "tokens_per_sec": 2048.0},
        "initial_estimate_seconds": 123.5,
        "estimated_vs_actual_seconds": {"estimated_seconds": 123.5, "actual_seconds": 120.0},
    }

    path = save_checkpoint(
        model,
        optimizer,
        TinyScaler(),
        step=10,
        val_loss=1.5,
        cfg={"train": {"model_preset": "85M"}},
        out_dir=tmp_path,
        tag="final",
        provenance=provenance,
        training_metadata=training_metadata,
        final_step=10,
    )

    manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"] == provenance

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["training_metadata"]
    assert metadata["hardware_profile"]["gpu_memory_gb"] == 3.99
    assert metadata["requested_configuration"]["model_preset"] == "117M"
    assert metadata["effective_configuration"]["model_preset"] == "85M"
    assert metadata["preflight"]["tokens_per_sec"] == 2048.0
    assert metadata["estimated_vs_actual_seconds"]["actual_seconds"] == 120.0


def test_capability_discovery_records_verified_and_failed_probe(monkeypatch):
    import pytest
    torch = pytest.importorskip('torch')
    from pipeline import preflight
    from pipeline.model_sizer import HardwareProfile

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x, y):
            loss = (self.weight * (x.float().mean() - y.float().mean()) * 0 + self.weight.pow(2)).mean()
            return None, loss

    def make_model(cfg):
        if int(cfg.seq_len) == 512:
            raise RuntimeError("synthetic probe failure")
        return TinyModel()

    monkeypatch.setattr(preflight, "LlamaModel", make_model)
    hw = HardwareProfile("test", 4, 8.0, 0, 0.0, None, False)
    report = preflight.discover_capabilities(
        hw,
        candidates=(("85M", 256), ("117M", 512)),
        benchmark_steps=1,
        vocab_size=32,
    )
    assert report["tested_combinations"] == 2
    assert report["results"][0]["viable"] is True
    assert report["results"][1]["viable"] is False
    assert "synthetic probe failure" in report["results"][1]["failure"]


def test_model_budget_generates_real_intermediate_architectures():
    mid = ModelConfig.from_target_params(220_000_000, seq_len=512)
    assert 117_000_000 < mid.param_count() < 360_000_000
    assert mid.seq_len == 512
    assert mid.n_layers > 0
    assert mid.d_model % mid.n_heads == 0


def test_model_budget_preserves_named_preset_geometry_as_nearby_search_points():
    for target in (85_000_000, 117_000_000, 360_000_000):
        cfg = ModelConfig.from_target_params(target)
        assert cfg.param_count() > 0
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads == cfg.n_kv_heads


def test_capability_report_marks_verified_and_failed_results_with_architecture(monkeypatch):
    import pytest
    torch = pytest.importorskip('torch')
    from pipeline import preflight
    from pipeline.model_sizer import HardwareProfile

    class TinyModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x, y):
            loss = (self.weight * (x.float().mean() - y.float().mean()) * 0 + self.weight.pow(2)).mean()
            return None, loss

    def fake_model(cfg):
        if cfg.seq_len >= 512:
            raise RuntimeError("synthetic context failure")
        return TinyModel(cfg)

    monkeypatch.setattr(preflight, "LlamaModel", fake_model)
    hw = HardwareProfile("test", 4, 8.0, 0, 0.0, None, False)
    report = preflight.discover_capabilities(
        hw,
        candidates=(("85M", 256), ("117M", 512)),
        benchmark_steps=1,
        vocab_size=32,
    )
    assert report["schema"] == 2
    assert report["verified_combinations"] == 1
    assert report["failed_combinations"] == 1
    assert report["results"][0]["architecture"]["d_model"] == 640
    assert report["results"][1]["viable"] is False

def test_duration_counts_events_at_final_step():
    estimate = estimate_duration(
        total_steps=100,
        batch_size=1,
        grad_accum_steps=1,
        seq_len=10,
        tokens_per_sec=10.0,
        eval_every_steps=20,
        eval_batches=2,
        checkpoint_every_steps=50,
    )
    assert estimate["estimated_eval_count"] == 6
    assert estimate["estimated_checkpoint_count"] == 8
    assert [item["step"] for item in estimate["estimated_eval_times"] if item.get("kind") != "final"] == [
        20, 40, 60, 80, 100
    ]
    assert [item["step"] for item in estimate["estimated_checkpoint_times"] if item.get("kind") == "scheduled"] == [
        50, 100
    ]


def test_duration_milestones_include_configured_overhead_in_trainer_order():
    estimate = estimate_duration(
        total_steps=10,
        batch_size=1,
        grad_accum_steps=1,
        seq_len=10,
        tokens_per_sec=10.0,
        eval_every_steps=5,
        eval_batches=1,
        checkpoint_every_steps=10,
        eval_seconds=2.0,
        checkpoint_seconds=3.0,
    )
    assert estimate["estimated_eval_count"] == 3
    assert estimate["estimated_checkpoint_count"] == 4
    assert estimate["estimated_eval_times"][-1]["estimated_elapsed_seconds"] == 25.0
    assert estimate["estimated_checkpoint_times"][-1]["estimated_elapsed_seconds"] == 28.0
    assert estimate["estimated_duration_seconds"] == 28.0


def test_duration_marks_missing_event_overhead_incomplete():
    estimate = estimate_duration(
        total_steps=10,
        batch_size=1,
        grad_accum_steps=1,
        seq_len=10,
        tokens_per_sec=10.0,
        eval_every_steps=5,
        eval_batches=1,
        checkpoint_every_steps=10,
    )
    assert estimate["overhead_estimate_complete"] is False
    assert estimate["missing_overhead_measurements"] == ["evaluation", "checkpoint"]


def test_preflight_calibrates_evaluation_and_checkpoint_overhead(monkeypatch, tmp_path):
    import pytest
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from pipeline import preflight

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.cfg = SimpleNamespace(seq_len=4)

        def forward(self, x, y):
            return None, self.weight.square() + (x.float().mean() - y.float().mean()) * 0

    class TinyLoader:
        def __init__(self, _directory, _split, seq_len, **_kwargs):
            self.seq_len = seq_len

        def next_batch(self, batch_size):
            values = torch.ones((batch_size, self.seq_len), dtype=torch.long)
            return values, values

    monkeypatch.setattr(preflight, "_make_model", lambda _cfg: TinyModel())
    monkeypatch.setattr(preflight, "ShardDataLoader", TinyLoader)
    result = preflight.run_preflight(
        {
            "seq_len": 4,
            "batch_size": 1,
            "grad_accum_steps": 1,
            "eval_batches": 1,
            "shard_dtype": "uint16",
        },
        tmp_path,
        warmup_steps=0,
        benchmark_steps=1,
    )
    assert result.viable is True
    assert result.eval_seconds_per_event is not None
    assert result.eval_seconds_per_event >= 0
    assert result.checkpoint_seconds_per_event is not None
    assert result.checkpoint_seconds_per_event > 0
