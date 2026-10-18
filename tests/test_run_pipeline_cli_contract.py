import logging
import sys

import run_pipeline


def test_cli_log_level_updates_root_logger(monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(logging.getLogger(), "setLevel", observed.append)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--list-stages", "--log-level", "DEBUG"])

    assert run_pipeline.main() == 0
    assert observed == [logging.DEBUG]
    assert "Available stages:" in capsys.readouterr().out
