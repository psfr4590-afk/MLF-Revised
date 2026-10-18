from __future__ import annotations

from types import SimpleNamespace

import bootstrap


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def __mul__(self, multiplier):
        return FakeTensor([value * multiplier for value in self.values])

    def sum(self):
        return FakeTensor([sum(self.values)])

    def item(self):
        return self.values[0]


def _fake_torch(device_calls, result_factory=FakeTensor):
    backend = SimpleNamespace(synchronize=lambda: None)
    return SimpleNamespace(
        float32="float32",
        cuda=backend,
        xpu=backend,
        mps=backend,
        device=lambda name: device_calls.append(name) or name,
        tensor=lambda values, dtype, device: result_factory(values),
    )


def test_tensor_smoke_uses_cuda_when_available(monkeypatch):
    calls = []
    fake = _fake_torch(calls)
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: fake)

    ok, detail = bootstrap._torch_tensor_smoke({"cuda_available": True})

    assert ok
    assert calls == ["cuda"]
    assert "tensor arithmetic PASS" in detail


def test_tensor_smoke_selects_xpu_mps_or_cpu(monkeypatch):
    for runtime, expected in (
        ({"xpu_available": True}, "xpu"),
        ({"mps_available": True}, "mps"),
        ({}, "cpu"),
    ):
        calls = []
        fake = _fake_torch(calls)
        monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name, f=fake: f)

        ok, _ = bootstrap._torch_tensor_smoke(runtime)

        assert ok
        assert calls == [expected]


def test_tensor_smoke_fails_on_wrong_result(monkeypatch):
    calls = []
    fake = _fake_torch(calls, lambda values: FakeTensor([99]))
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: fake)

    ok, detail = bootstrap._torch_tensor_smoke({})

    assert not ok
    assert "unexpected arithmetic result" in detail


def test_tensor_smoke_reports_device_operation_failure(monkeypatch):
    def fail_tensor(*args, **kwargs):
        raise RuntimeError("device kernel failed")

    fake = SimpleNamespace(
        float32="float32",
        device=lambda name: name,
        tensor=fail_tensor,
    )
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: fake)

    ok, detail = bootstrap._torch_tensor_smoke({"cuda_available": True})

    assert not ok
    assert "RuntimeError: device kernel failed" in detail
