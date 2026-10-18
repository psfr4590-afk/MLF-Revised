from fastapi.testclient import TestClient

from command_center.web import app


def test_command_center_initializes_during_lifespan(monkeypatch):
    import command_center.web as web

    calls = []
    monkeypatch.setattr(web, "init", lambda: calls.append("init"))

    with TestClient(app):
        assert calls == ["init"]


def test_command_center_has_no_deprecated_startup_event():
    import inspect
    import command_center.web as web

    source = inspect.getsource(web)
    assert "@app.on_event(\"startup\")" not in source
    assert "lifespan=lifespan" in source
