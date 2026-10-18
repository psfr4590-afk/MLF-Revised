import logging
import sys
from pathlib import Path

import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_list_groups_does_not_construct_pipeline(monkeypatch, capsys):
    class ExplodingPipeline:
        def __init__(self, *args, **kwargs):
            raise AssertionError("--list-groups must not construct Pipeline")

    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--list-groups"])
    monkeypatch.setitem(run_pipeline.__dict__, "_Pipeline_guard", ExplodingPipeline)

    import pipeline.orchestrator
    monkeypatch.setattr(pipeline.orchestrator, "Pipeline", ExplodingPipeline)

    assert run_pipeline.main() == 0
    output = capsys.readouterr().out
    assert "Configured dataset groups:" in output
    assert "smoke" in output
    assert "Bounded local smoke corpus" in output


def test_pipeline_log_level_is_reapplied_after_pipeline_initialization(monkeypatch):
    class FakePipeline:
        def __init__(self, *args, **kwargs):
            logging.getLogger().setLevel(logging.INFO)

        def run(self, *args, **kwargs):
            return None

        def close(self):
            return None

    import pipeline.orchestrator
    monkeypatch.setattr(pipeline.orchestrator, "Pipeline", FakePipeline)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--log-level", "DEBUG"])

    run_pipeline.main()
    assert logging.getLogger().level == logging.DEBUG


def test_invalid_stages_do_not_construct_pipeline(monkeypatch, capsys):
    class ExplodingPipeline:
        def __init__(self, *args, **kwargs):
            raise AssertionError("invalid --stages must be rejected before Pipeline startup")

    import pipeline.orchestrator
    monkeypatch.setattr(pipeline.orchestrator, "Pipeline", ExplodingPipeline)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--stages", "crawl,not-a-stage"])

    assert run_pipeline.main() == 1
    output = capsys.readouterr().out
    assert "Unknown stages" in output
    assert "not-a-stage" in output


def test_version_is_stable_and_does_not_start_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--version"])

    try:
        run_pipeline.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--version should terminate through argparse")

    assert capsys.readouterr().out.strip() == "Model Lab 1.3.0rc1"


def test_hardware_report_requires_doctor(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--hardware-report"])

    assert run_pipeline.main() == 1
    assert "requires --doctor" in capsys.readouterr().out


def test_help_exposes_canonical_examples(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--help"])

    try:
        run_pipeline.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--help should terminate through argparse")

    output = capsys.readouterr().out
    assert "Model Lab 1.3.0rc1" in output
    assert "--no-resume" in output
    assert "python run_pipeline.py --doctor" in output
    assert "mlab --help" in output
