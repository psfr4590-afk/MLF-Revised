from pathlib import Path

import yaml

from pipeline.doctor import run_doctor
from pipeline.config_validation import validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_is_non_destructive_and_reports_required_checks():
    before = {p.relative_to(ROOT) for p in ROOT.rglob("pipeline/doctor.py")}
    ok, checks = run_doctor(ROOT)
    assert isinstance(ok, bool)
    assert any(item["name"] == "Python" for item in checks)
    after = {p.relative_to(ROOT) for p in ROOT.rglob("pipeline/doctor.py")}
    assert before == after


def test_main_config_has_one_model_preset_source():
    cfg = yaml.safe_load((ROOT / "config/pipeline_config.yaml").read_text(encoding="utf-8"))
    assert "model_preset" not in cfg["pipeline"]
    assert cfg["train"]["model_preset"] in {"85M", "117M", "360M"}


def test_smoke_config_passes_validator():
    cfg = yaml.safe_load((ROOT / "config/pipeline_config.smoke.yaml").read_text(encoding="utf-8"))
    validate_config(cfg)


def test_smoke_group_is_bounded():
    cfg = yaml.safe_load((ROOT / "config/dataset_groups.smoke.yaml").read_text(encoding="utf-8"))
    group = cfg["dataset_groups"][0]
    assert group["sources"]["github"] is False
    assert group["sources"]["arxiv"] is False
    assert group["sources"]["huggingface"] is False
    assert group["sources"]["google"] is False
