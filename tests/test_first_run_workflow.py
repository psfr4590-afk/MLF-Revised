from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _load_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def test_default_runner_and_docs_share_one_starter_profile():
    runner = _read("run_pipeline.py")
    readme = _read("README.md")
    start_here = _read("docs/development/START_HERE.md")

    assert 'DEFAULT_CONFIG = "config/pipeline_config.yaml"' in runner
    assert 'parser.add_argument("--config", default=DEFAULT_CONFIG' in runner
    assert "python .\\run_pipeline.py --no-resume" in readme
    assert "python3 run_pipeline.py --no-resume" in readme
    assert "python .\\run_pipeline.py --no-resume" in start_here
    assert "python3 run_pipeline.py --no-resume" in start_here

    for doc in (readme, start_here):
        assert "pipeline_config.smoke.yaml" not in doc.split("## Bounded verification profile", 1)[0]


def test_canonical_starter_is_bounded_and_verification_profile_is_secondary():
    starter = _load_config("pipeline_config.yaml")
    verification = _load_config("pipeline_config.smoke.yaml")

    assert starter["train"]["total_steps"] <= 10
    assert starter["train"]["allow_cpu_training"] is True
    assert starter["stages"]["export"] is False
    assert starter["crawl"]["max_pages_per_domain"] <= 2

    assert verification["train"]["total_steps"] <= 10
    assert verification["train"]["allow_cpu_training"] is True
