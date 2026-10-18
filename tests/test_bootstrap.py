from __future__ import annotations

import bootstrap


def test_python_requirement_message_is_current() -> None:
    message = bootstrap._python_requirement_message()
    assert "Python 3.11-3.14 required" in message
    assert "lxml 5.x" not in message


def test_python_requirement_message_does_not_claim_old_lxml_reason() -> None:
    message = bootstrap._python_requirement_message()
    assert "Python 3.15+ is not supported" in message
    assert "dependency compatibility" in message


def test_hardware_profile_defaults_to_unknown() -> None:
    profile = bootstrap.HardwareProfile()
    assert profile.ram_gib is None
    assert profile.gpu_name is None
    assert profile.gpu_vram_gib is None
    assert profile.nvidia_smi_available is False


def test_torch_channel_summary_cpu() -> None:
    profile = bootstrap.HardwareProfile(ram_gib=8.0, gpu_name="NVIDIA GeForce GTX 1650", gpu_vram_gib=4.0)
    summary = bootstrap.torch_channel_summary("cpu", profile)
    assert summary["selected_channel"] == "cpu"
    assert summary["gpu_detected"] is True
    assert summary["cuda_install_requested"] is False


def test_torch_channel_summary_default_is_explicit() -> None:
    profile = bootstrap.HardwareProfile(ram_gib=8.0, gpu_name="NVIDIA GeForce GTX 1650", gpu_vram_gib=4.0)
    summary = bootstrap.torch_channel_summary("default", profile)
    assert summary["selected_channel"] == "default"
    assert summary["gpu_detected"] is True
    assert summary["cuda_install_requested"] is False
    assert summary["hardware_warning"] is not None


def test_torch_channel_summary_cuda() -> None:
    profile = bootstrap.HardwareProfile(ram_gib=8.0, gpu_name="NVIDIA GeForce GTX 1650", gpu_vram_gib=4.0)
    summary = bootstrap.torch_channel_summary("cuda", profile)
    assert summary["selected_channel"] == "cuda"
    assert summary["gpu_detected"] is True
    assert summary["cuda_install_requested"] is True
    assert summary["hardware_warning"] is None


def test_hardware_profile_json_is_stable() -> None:
    profile = bootstrap.HardwareProfile(
        ram_gib=8.0,
        cpu_name="AMD Ryzen 5 4600H with Radeon Graphics",
        cpu_cores=6,
        cpu_threads=12,
        gpu_name="NVIDIA GeForce GTX 1650",
        gpu_vram_gib=4.0,
        nvidia_smi_available=True,
    )
    assert profile.as_dict() == {
        "ram_gib": 8.0,
        "cpu_name": "AMD Ryzen 5 4600H with Radeon Graphics",
        "cpu_cores": 6,
        "cpu_threads": 12,
        "gpu_name": "NVIDIA GeForce GTX 1650",
        "gpu_vram_gib": 4.0,
        "gpu_vendor": "nvidia",
        "nvidia_smi_available": True,
        "nvidia_smi_query_succeeded": False,
    }


def test_bootstrap_required_runtime_dependencies_include_semantic_dedup() -> None:
    assert "sentence_transformers" in bootstrap.REQUIRED
    assert "faiss" in bootstrap.REQUIRED


def test_torch_variant_matches_cuda_and_cpu(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "_installed_torch_version", lambda: "2.11.0+cu128")
    assert bootstrap._torch_variant_matches("cuda")
    assert not bootstrap._torch_variant_matches("cpu")

    monkeypatch.setattr(bootstrap, "_installed_torch_version", lambda: "2.14.1+cpu")
    assert not bootstrap._torch_variant_matches("cuda")
    assert bootstrap._torch_variant_matches("cpu")


def test_torch_variant_matches_missing(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "_installed_torch_version", lambda: None)
    assert not bootstrap._torch_variant_matches("cuda")


def test_validate_cuda_torch_rejects_cpu_variant(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "_installed_torch_version", lambda: "2.14.1+cpu")
    assert bootstrap._validate_requested_torch_channel("cuda") == 2


def test_validate_cpu_torch_rejects_cuda_variant(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "_installed_torch_version", lambda: "2.11.0+cu128")
    assert bootstrap._validate_requested_torch_channel("cpu") == 2


def test_doctor_allows_cpu_fallback_when_cuda_is_unavailable(monkeypatch, capsys) -> None:
    profile = bootstrap.HardwareProfile(gpu_name="NVIDIA GeForce GTX 1650", gpu_vram_gib=4.0)
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: profile)
    monkeypatch.setattr(
        bootstrap,
        "torch_runtime_status",
        lambda: {
            "installed": True,
            "importable": True,
            "version": "2.14.1+cpu",
            "cuda_available": False,
            "cuda_version": None,
            "hip_available": False,
            "xpu_available": False,
            "mps_available": False,
            "probe_errors": [],
            "device_count": 0,
        },
    )
    monkeypatch.setattr(bootstrap, "_importable", lambda name: True)
    monkeypatch.setattr(bootstrap.RECONCILER.__class__, "is_file", lambda self: True)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda exe: "present")
    assert bootstrap.doctor() == 0
    output = capsys.readouterr().out
    assert "Verified runtime recommendation: cpu (CUDA unavailable)" in output
    assert "Bootstrap doctor: PASS" in output


def test_doctor_fails_when_tensor_operation_fails(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    profile = bootstrap.HardwareProfile(gpu_name="Generic CPU test host")
    runtime = {
        "installed": True,
        "importable": True,
        "version": "2.14.1+cpu",
        "cuda_available": False,
        "cuda_version": None,
        "hip_available": False,
        "xpu_available": False,
        "mps_available": False,
        "probe_errors": [],
        "device_count": 0,
    }

    def fail_tensor(*args, **kwargs):
        raise RuntimeError("deliberate tensor smoke failure")

    fake_torch = SimpleNamespace(
        device=lambda name: name,
        float32="float32",
        tensor=fail_tensor,
    )
    original_import_module = bootstrap.importlib.import_module

    def import_module(name, package=None):
        if name == "torch":
            return fake_torch
        return original_import_module(name, package)

    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: profile)
    monkeypatch.setattr(bootstrap, "torch_runtime_status", lambda: runtime)
    monkeypatch.setattr(bootstrap.importlib, "import_module", import_module)
    monkeypatch.setattr(bootstrap, "_importable", lambda name: True)
    monkeypatch.setattr(bootstrap.RECONCILER.__class__, "is_file", lambda self: True)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda exe: "present")

    assert bootstrap.doctor() == 2
    output = capsys.readouterr().out
    assert "Torch tensor smoke: FAIL" in output
    assert "deliberate tensor smoke failure" in output
    assert "Bootstrap doctor: FAIL" in output
