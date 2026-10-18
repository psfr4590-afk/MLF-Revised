from __future__ import annotations

from pipeline.provenance import training_compatibility_config


def test_resume_controls_do_not_change_checkpoint_compatibility_identity():
    baseline = {
        "model_preset": "85M", "vocab_size": 512, "seq_len": 32,
        "lr_max": 3e-4, "resume": False, "total_steps": 2,
    }
    resumed = {**baseline, "resume": True, "total_steps": 4}
    assert training_compatibility_config(baseline) == training_compatibility_config(resumed)


def test_model_or_optimizer_semantics_do_change_checkpoint_compatibility_identity():
    baseline = {"model_preset": "85M", "vocab_size": 512, "seq_len": 32, "lr_max": 3e-4}
    changed = {**baseline, "seq_len": 64}
    assert training_compatibility_config(baseline) != training_compatibility_config(changed)
