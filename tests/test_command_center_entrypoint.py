from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_browser_flag_controls_browser_launch():
    source = (ROOT / "run_command_center.py").read_text(encoding="utf-8")
    assert "--no-browser" in source
    assert "if not args.no_browser" in source
    assert "webbrowser.open" in source
