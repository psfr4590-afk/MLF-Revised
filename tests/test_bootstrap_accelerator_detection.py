from __future__ import annotations

import pytest

import bootstrap


def test_hardware_profile_recognizes_nvidia_from_adapter_name():
    profile = bootstrap.HardwareProfile(gpu_name="NVIDIA GeForce GTX 1650")

    assert profile.nvidia_gpu_detected
    assert profile.gpu_detected


def test_hardware_profile_recognizes_nvidia_from_adapter_name_even_if_vendor_is_unknown():
    profile = bootstrap.HardwareProfile(
        gpu_name="NVIDIA GeForce GTX 1650",
        gpu_vendor="unknown",
    )

    assert profile.nvidia_gpu_detected


def test_hardware_profile_recognizes_nvidia_from_nvidia_smi_when_name_omits_vendor():
    profile = bootstrap.HardwareProfile(
        gpu_name="Tesla T4",
        nvidia_smi_query_succeeded=True,
    )

    assert profile.nvidia_gpu_detected
    assert profile.gpu_detected


def test_hardware_profile_recognizes_amd_and_intel_adapters():
    amd = bootstrap.HardwareProfile(
        gpu_name="AMD Radeon Graphics",
        gpu_vendor=bootstrap._gpu_vendor("AMD Radeon Graphics"),
    )
    intel = bootstrap.HardwareProfile(
        gpu_name="Intel Arc Graphics",
        gpu_vendor=bootstrap._gpu_vendor("Intel Arc Graphics"),
    )

    assert amd.gpu_vendor == "amd"
    assert intel.gpu_vendor == "intel"
    assert not amd.nvidia_gpu_detected
    assert not intel.nvidia_gpu_detected


def test_runtime_recommendation_uses_verified_cuda_not_gpu_presence_alone():
    profile = bootstrap.HardwareProfile(gpu_name="NVIDIA GeForce GTX 1650")
    runtime = {"installed": True, "importable": True, "cuda_available": False}

    # Doctor may recommend CPU as a diagnosed fallback; auto-install still refuses silent fallback.
    assert bootstrap.recommended_torch_channel(profile, runtime) == "cpu (CUDA unavailable)"


def test_runtime_recommendation_allows_verified_cuda():
    profile = bootstrap.HardwareProfile(gpu_name="NVIDIA GeForce GTX 1650")
    runtime = {"installed": True, "importable": True, "cuda_available": True}

    assert bootstrap.recommended_torch_channel(profile, runtime) == "cuda (usable)"


def test_runtime_recommendation_keeps_verified_intel_xpu():
    profile = bootstrap.HardwareProfile(
        gpu_name="Intel Arc Graphics",
        gpu_vendor="intel",
    )
    runtime = {"installed": True, "importable": True, "xpu_available": True}

    assert bootstrap.recommended_torch_channel(profile, runtime) == "xpu (usable)"


def test_runtime_recommendation_does_not_silently_fall_back_for_amd_gpu():
    profile = bootstrap.HardwareProfile(
        gpu_name="AMD Radeon Graphics",
        gpu_vendor="amd",
    )
    runtime = {"installed": False, "importable": False}

    assert bootstrap.recommended_torch_channel(profile, runtime).startswith(
        "manual review required"
    )


def test_runtime_recommendation_uses_cpu_when_no_gpu_is_detected():
    profile = bootstrap.HardwareProfile()
    runtime = {"installed": False, "importable": False}

    assert bootstrap.recommended_torch_channel(profile, runtime) == "cpu"


def test_runtime_dependency_inventory_comes_from_pyproject_metadata():
    names = {name.lower() for name in bootstrap._declared_runtime_distributions()}

    assert "pyyaml" in names
    assert "requests" in names
    assert "fastapi" in names
    assert "faiss-cpu" in names
    assert "torch" not in names

def test_runtime_recommendation_keeps_verified_amd_rocm():
    profile = bootstrap.HardwareProfile(
        gpu_name="AMD Radeon Graphics",
        gpu_vendor="amd",
    )
    runtime = {"installed": True, "importable": True, "cuda_available": True, "hip_available": True}

    assert bootstrap.recommended_torch_channel(profile, runtime) == "rocm (usable)"

def test_every_declared_runtime_distribution_has_an_import_probe():
    declared = {
        name.lower().replace("_", "-")
        for name in bootstrap._declared_runtime_distributions()
    }

    assert declared - set(bootstrap.DISTRIBUTION_IMPORTS) == set()
    assert "torch" not in declared

def test_auto_install_selects_rocm_for_amd_gpu_on_linux(monkeypatch):
    profile = bootstrap.HardwareProfile(
        gpu_name="AMD Radeon Graphics",
        gpu_vendor="amd",
    )
    calls = []
    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: profile)
    monkeypatch.setattr(
        bootstrap,
        "torch_runtime_status",
        lambda: {"installed": False, "importable": False, "hip_available": False},
    )
    monkeypatch.setattr(bootstrap, "_remove_incompatible_torch", lambda channel: calls.append(("remove", channel)) or 0)
    monkeypatch.setattr(bootstrap, "_pip_install", lambda requirements, *extra: calls.append((str(requirements), *extra)) or 0)
    monkeypatch.setattr(bootstrap, "_validate_requested_torch_channel", lambda channel: calls.append(("validate", channel)) or 0)

    assert bootstrap.install("auto") == 0
    assert ("remove", "rocm") in calls
    assert (str(bootstrap.TORCH_REQUIREMENTS), "--index-url", bootstrap.ROCM_TORCH_INDEX) in calls
    assert ("validate", "rocm") in calls


def test_auto_install_selects_xpu_for_intel_gpu(monkeypatch):
    profile = bootstrap.HardwareProfile(
        gpu_name="Intel Arc Graphics",
        gpu_vendor="intel",
    )
    calls = []
    monkeypatch.setattr(bootstrap.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: profile)
    monkeypatch.setattr(
        bootstrap,
        "torch_runtime_status",
        lambda: {"installed": False, "importable": False, "xpu_available": False},
    )
    monkeypatch.setattr(bootstrap, "_remove_incompatible_torch", lambda channel: calls.append(("remove", channel)) or 0)
    monkeypatch.setattr(bootstrap, "_pip_install", lambda requirements, *extra: calls.append((str(requirements), *extra)) or 0)
    monkeypatch.setattr(bootstrap, "_validate_requested_torch_channel", lambda channel: calls.append(("validate", channel)) or 0)

    assert bootstrap.install("auto") == 0
    assert ("remove", "xpu") in calls
    assert (str(bootstrap.TORCH_REQUIREMENTS), "--index-url", bootstrap.XPU_TORCH_INDEX) in calls
    assert ("validate", "xpu") in calls

def test_explicit_gpu_runtime_is_rejected_without_matching_hardware(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: bootstrap.HardwareProfile())
    install_calls = []
    monkeypatch.setattr(bootstrap, "_pip_install", lambda *args: install_calls.append(args) or 0)

    assert bootstrap.install("cuda") == 2
    assert install_calls == []
    assert "no compatible GPU/platform was detected" in capsys.readouterr().err


def test_runtime_status_catches_cuda_device_count_errors(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            raise RuntimeError("driver enumeration failed")

    class FakeTorch:
        __version__ = "test"
        cuda = FakeCuda()
        version = type("Version", (), {"cuda": "12.8", "hip": None})()
        xpu = type("Xpu", (), {"is_available": staticmethod(lambda: False)})()
        backends = type("Backends", (), {"mps": type("Mps", (), {"is_available": staticmethod(lambda: False)})()})()

    monkeypatch.setattr(bootstrap.importlib.metadata, "version", lambda name: "2.11.0")
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: FakeTorch())
    result = bootstrap.torch_runtime_status()

    assert result["cuda_available"] is False
    assert result["device_count"] == 0
    assert any("device_count" in error and "driver enumeration failed" in error for error in result["probe_errors"])


def test_requirement_validation_checks_version_constraints():
    assert bootstrap._requirement_satisfied("numpy>=1.26,<3", "2.0.0")
    assert not bootstrap._requirement_satisfied("numpy>=1.26,<3", "3.0.0")
    assert not bootstrap._requirement_satisfied("fastapi>=0.141.1,<1", "0.100.0")


def test_source_import_audit_flags_unmapped_third_party_import(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    (tmp_path / "sample.py").write_text(
        "import json\nimport torch\nimport unlisted_external_package\n",
        encoding="utf-8",
    )

    issues = bootstrap._source_import_dependency_issues({"torch"})

    assert "unmapped third-party source import: unlisted_external_package" in issues
    assert not any("json" in issue for issue in issues)
    assert not any("torch" in issue for issue in issues)


def test_source_import_audit_accepts_declared_distribution(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    (tmp_path / "sample.py").write_text("import json\nimport numpy\n", encoding="utf-8")
    vendor = tmp_path / "third_party" / "llama.cpp" / "tools"
    vendor.mkdir(parents=True)
    (vendor / "vendor_tool.py").write_text("import pydantic_models_to_grammar\nimport PySide6\n", encoding="utf-8")

    assert bootstrap._source_import_dependency_issues({"numpy"}) == []




def test_runtime_status_preserves_cuda_probe_error_field(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            raise OSError("CUDA runtime probe blocked")

    class FakeTorch:
        __version__ = "test"
        cuda = FakeCuda()
        version = type("Version", (), {"cuda": "12.8", "hip": None})()
        xpu = type("Xpu", (), {"is_available": staticmethod(lambda: False)})()
        backends = type("Backends", (), {"mps": type("Mps", (), {"is_available": staticmethod(lambda: False)})()})()

    monkeypatch.setattr(bootstrap.importlib.metadata, "version", lambda name: "2.11.0")
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: FakeTorch())
    result = bootstrap.torch_runtime_status()

    assert result["cuda_available"] is False
    assert result["device_count"] == 0
    assert "CUDA runtime probe blocked" in result["cuda_error"]


def test_source_import_audit_ignores_isolated_ci_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    (tmp_path / "sample.py").write_text("import json\nimport numpy\n", encoding="utf-8")
    generated_package = tmp_path / ".ci-venv" / "lib" / "site-packages" / "unrelated_vendor_pkg"
    generated_package.mkdir(parents=True)
    (generated_package / "__init__.py").write_text(
        "import dependency_that_is_not_part_of_this_project\n", encoding="utf-8"
    )

    assert bootstrap._source_import_dependency_issues({"numpy"}) == []


def test_security_tool_is_declared_as_optional_dependency():
    optional = {name.lower().replace("_", "-") for name in bootstrap._declared_optional_runtime_distributions()}

    assert "pip-audit" in optional
    assert bootstrap.DISTRIBUTION_IMPORTS["pip-audit"] == "pip_audit"


def test_runtime_recommendation_covers_unavailable_and_cpu_only_states():
    gpu = bootstrap.HardwareProfile(gpu_name="NVIDIA GeForce GTX 1650")
    no_gpu = bootstrap.HardwareProfile()

    assert bootstrap.recommended_torch_channel(
        gpu, {"importable": False, "cuda_available": False}
    ).startswith("manual review required")
    assert bootstrap.recommended_torch_channel(
        no_gpu, {"importable": True, "cuda_available": False}
    ) == "cpu"


def test_torch_runtime_status_reports_installed_but_blocked_import(monkeypatch):
    def blocked_import(name):
        if name == "torch":
            raise OSError("application control blocked torch DLL")
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(bootstrap.importlib, "import_module", blocked_import)
    monkeypatch.setattr(bootstrap.importlib.metadata, "version", lambda name: "2.11.0+cu128")

    status = bootstrap.torch_runtime_status()

    assert status["installed"] is True
    assert status["importable"] is False
    assert status["version"] == "2.11.0+cu128"
    assert "application control blocked torch DLL" in status["error"]


def test_torch_runtime_status_preserves_cuda_probe_failure(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            raise OSError("CUDA runtime probe blocked")

    class FakeTorch:
        __version__ = "2.11.0+cu128"
        cuda = FakeCuda()
        version = type("Version", (), {"cuda": "12.8", "hip": None})()

    monkeypatch.setattr(bootstrap.importlib.metadata, "version", lambda name: "2.11.0+cu128")
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: FakeTorch())

    status = bootstrap.torch_runtime_status()

    assert status["installed"] is True
    assert status["importable"] is True
    assert status["cuda_available"] is False
    assert "CUDA runtime probe blocked" in status["cuda_error"]


def test_cuda_channel_validation_preserves_probe_failure_detail(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "_installed_torch_version", lambda: "2.11.0+cu128")
    monkeypatch.setattr(
        bootstrap,
        "torch_runtime_status",
        lambda: {
            "installed": True,
            "importable": True,
            "cuda_available": False,
            "cuda_error": "OSError: CUDA runtime probe blocked",
        },
    )

    assert bootstrap._validate_requested_torch_channel("cuda") == 2
    assert "CUDA runtime probe blocked" in capsys.readouterr().err


def test_cpu_channel_validation_rejects_unimportable_torch(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "_installed_torch_version", lambda: "2.14.1+cpu")
    monkeypatch.setattr(
        bootstrap,
        "torch_runtime_status",
        lambda: {
            "installed": True,
            "importable": False,
            "cuda_available": False,
            "error": "OSError: blocked DLL",
        },
    )

    assert bootstrap._validate_requested_torch_channel("cpu") == 2
    assert "blocked DLL" in capsys.readouterr().err


def test_doctor_reports_cuda_probe_error_and_keeps_cpu_fallback(monkeypatch, capsys):
    profile = bootstrap.HardwareProfile(gpu_name="NVIDIA GeForce GTX 1650", gpu_vram_gib=4.0)
    monkeypatch.setattr(bootstrap, "detect_hardware", lambda: profile)
    monkeypatch.setattr(
        bootstrap,
        "torch_runtime_status",
        lambda: {
            "installed": True,
            "importable": True,
            "version": "2.11.0+cu128",
            "cuda_available": False,
            "cuda_version": "12.8",
            "hip_available": False,
            "xpu_available": False,
            "mps_available": False,
            "device_count": 0,
            "cuda_error": "OSError: CUDA runtime probe blocked",
            "probe_errors": ["cuda: OSError: CUDA runtime probe blocked"],
        },
    )
    monkeypatch.setattr(bootstrap, "_importable", lambda name: True)
    monkeypatch.setattr(bootstrap.RECONCILER.__class__, "is_file", lambda self: True)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda exe: "present")

    assert bootstrap.doctor() == 0
    output = capsys.readouterr().out
    assert "Accelerator probe error: cuda: OSError: CUDA runtime probe blocked" in output
    assert "Verified runtime recommendation: cpu (CUDA unavailable)" in output
    assert "Bootstrap doctor: PASS" in output


def test_nvidia_smi_executable_without_successful_query_does_not_prove_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(bootstrap, "_windows_cpu", lambda: ("test CPU", 4, 8))
    monkeypatch.setattr(bootstrap, "_ram_gib", lambda: 8.0)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(bootstrap, "_run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "_generic_gpu", lambda: "AMD Radeon Graphics")

    profile = bootstrap.detect_hardware()

    assert profile.nvidia_smi_available
    assert not profile.nvidia_smi_query_succeeded
    assert not profile.nvidia_gpu_detected
    assert profile.gpu_vendor == "amd"


def test_nvidia_smi_query_success_identifies_tesla_t4_without_vendor_token(monkeypatch):
    monkeypatch.setattr(bootstrap, "_windows_cpu", lambda: ("test CPU", 4, 8))
    monkeypatch.setattr(bootstrap, "_ram_gib", lambda: 8.0)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(bootstrap, "_run_command", lambda *args, **kwargs: "Tesla T4, 16384")

    profile = bootstrap.detect_hardware()

    assert profile.gpu_name == "Tesla T4"
    assert profile.gpu_vram_gib == 16.0
    assert profile.nvidia_smi_available
    assert profile.nvidia_smi_query_succeeded
    assert profile.nvidia_gpu_detected
    assert profile.gpu_vendor == "nvidia"
    assert profile.as_dict()["gpu_vendor"] == "nvidia"


@pytest.mark.parametrize(
    ("name", "vendor", "runtime", "expected"),
    [
        ("NVIDIA Tesla T4", "nvidia", {"importable": True, "cuda_available": True}, "cuda (usable)"),
        ("AMD Radeon RX", "amd", {"importable": True, "cuda_available": True, "hip_available": True}, "rocm (usable)"),
        ("Intel Arc", "intel", {"importable": True, "xpu_available": True}, "xpu (usable)"),
        ("Apple M2", "apple", {"importable": True, "mps_available": True}, "mps (usable)"),
    ],
)
def test_runtime_recommendation_supports_each_declared_accelerator_backend(name, vendor, runtime, expected):
    profile = bootstrap.HardwareProfile(gpu_name=name, gpu_vendor=vendor)

    assert bootstrap.recommended_torch_channel(profile, runtime) == expected


def test_smoke_tries_an_alternate_accelerator_before_cpu(monkeypatch):
    class FakeTensor:
        def __init__(self, device):
            self.device = device

        def __mul__(self, value):
            if self.device == "cuda":
                raise RuntimeError("CUDA execution failed")
            return self

        def sum(self):
            return self

        def item(self):
            return 6.0

    class FakeBackend:
        @staticmethod
        def synchronize():
            return None

    class FakeTorch:
        float32 = "float32"
        cuda = FakeBackend()
        xpu = FakeBackend()
        mps = FakeBackend()

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def tensor(values, dtype, device):
            return FakeTensor(device)

    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: FakeTorch())

    passed, detail = bootstrap._torch_tensor_smoke(
        {"cuda_available": True, "xpu_available": True, "mps_available": False}
    )

    assert passed
    assert "cuda tensor operation failed" in detail
    assert "verified xpu fallback PASS" in detail
    assert "CPU fallback" not in detail


def test_accelerator_tensor_failure_is_reported_and_cpu_fallback_is_verified(monkeypatch):
    class FakeTensor:
        def __init__(self, device):
            self.device = device

        def __mul__(self, value):
            if self.device == "cuda":
                raise RuntimeError("simulated driver execution failure")
            return self

        def sum(self):
            return self

        def item(self):
            return 6.0

    class FakeCuda:
        @staticmethod
        def synchronize():
            raise RuntimeError("simulated driver synchronization failure")

    class FakeTorch:
        float32 = "float32"
        cuda = FakeCuda()

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def tensor(values, dtype, device):
            return FakeTensor(device)

    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: FakeTorch())

    passed, detail = bootstrap._torch_tensor_smoke(
        {"cuda_available": True, "xpu_available": False, "mps_available": False}
    )

    assert passed
    assert "cuda tensor operation failed" in detail
    assert "verified CPU fallback PASS" in detail
    assert "cpu tensor arithmetic PASS" in detail


@pytest.mark.parametrize(
    ("device_name", "runtime"),
    [
        ("cuda", {"cuda_available": True}),
        ("xpu", {"xpu_available": True}),
        ("mps", {"mps_available": True}),
        ("cpu", {}),
    ],
)
def test_tensor_smoke_runs_on_selected_backend(device_name, runtime, monkeypatch):
    class FakeTensor:
        def __mul__(self, value):
            return self

        def sum(self):
            return self

        def item(self):
            return 6.0

    class FakeBackend:
        @staticmethod
        def synchronize():
            return None

    class FakeTorch:
        float32 = "float32"
        cuda = FakeBackend()
        xpu = FakeBackend()
        mps = FakeBackend()

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def tensor(values, dtype, device):
            assert device == device_name
            return FakeTensor()

    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: FakeTorch())

    passed, detail = bootstrap._torch_tensor_smoke(runtime)

    assert passed
    assert f"{device_name} tensor arithmetic PASS" in detail
