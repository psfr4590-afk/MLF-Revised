import sys

import pytest

import scripts.verify_release as verify_release


def test_release_gate_does_not_claim_native_prerequisites_without_bootstrap(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["verify_release.py"])
    monkeypatch.setattr(verify_release, "run", lambda cmd: 0)

    assert verify_release.main() == 0
    output = capsys.readouterr().out
    assert "required environment checks are green" in output
    assert "Native artifact verification was not run" in output


def test_release_gate_bootstrap_mode_reports_native_check_without_claiming_unknown_evidence(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["verify_release.py", "--bootstrap-native"])
    commands = []

    def fake_run(cmd):
        commands.append(cmd)
        return 0

    monkeypatch.setattr(verify_release, "run", fake_run)
    monkeypatch.setattr(verify_release, "_native_smoke", lambda verify_ollama=False: 0)
    monkeypatch.setattr(verify_release, "NATIVE_EVIDENCE", {})

    assert verify_release.main() == 2
    output = capsys.readouterr().out
    assert "native export prerequisites are green" in output
    assert "native-smoke report contains a required non-PASS check" in output
    assert any("--ensure-llamacpp" in item for command in commands for item in command)


def test_rc_profile_requires_ui_probe(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["verify_release.py", "--profile", "rc", "--bootstrap-native", "--clean-clone"],
    )
    with pytest.raises(SystemExit) as exc_info:
        verify_release.main()
    assert exc_info.value.code != 0


def test_rc_profile_requires_clean_clone(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["verify_release.py", "--profile", "rc", "--bootstrap-native", "--ui-probe"],
    )
    with pytest.raises(SystemExit) as exc_info:
        verify_release.main()
    assert exc_info.value.code != 0
