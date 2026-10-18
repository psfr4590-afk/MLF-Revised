from __future__ import annotations

import json
from pathlib import Path

import scripts.system_info as system_info


def test_build_report_records_runtime_and_hardware(monkeypatch) -> None:
    monkeypatch.setattr(
        system_info.bootstrap,
        "detect_hardware",
        lambda: system_info.bootstrap.HardwareProfile(
            ram_gib=8.0,
            cpu_name="AMD Ryzen 5 4600H with Radeon Graphics",
            cpu_cores=6,
            cpu_threads=12,
            gpu_name="NVIDIA GeForce GTX 1650",
            gpu_vram_gib=4.0,
            nvidia_smi_available=True,
        ),
    )
    monkeypatch.setattr(
        system_info,
        "_torch_state",
        lambda: {
            "installed": True,
            "version": "2.14.0+cpu",
            "cuda_available": False,
            "torch_cuda_version": None,
            "device_count": 0,
            "devices": [],
        },
    )
    monkeypatch.setattr(
        system_info,
        "_nvidia_state",
        lambda: {
            "available": True,
            "gpus": [
                {"name": "NVIDIA GeForce GTX 1650", "driver_version": "32.0.15.9649", "vram_gib": 4.0}
            ],
        },
    )
    monkeypatch.setattr(system_info, "_pip_check", lambda: {"passed": True, "exit_code": 0, "output": "No broken requirements found."})

    report = system_info.build_report("cpu")

    assert report["schema_version"] == 1
    assert report["hardware"]["ram_gib"] == 8.0
    assert report["hardware"]["gpu_name"] == "NVIDIA GeForce GTX 1650"
    assert report["torch"]["version"] == "2.14.0+cpu"
    assert report["torch"]["cuda_available"] is False
    assert report["installation"]["selected_torch_channel"] == "cpu"
    assert report["installation"]["verified"] is True
    assert report["assessment"]["constrained_machine"] is True
    assert report["assessment"]["cuda_claim"] == "not_runtime_verified"


def test_write_report_is_atomic_json(tmp_path: Path) -> None:
    output = tmp_path / "runtime" / "system_info.json"
    report = {"schema_version": 1, "installation": {"verified": True}}

    system_info.write_report(report, output)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not output.with_suffix(".json.tmp").exists()


def test_cuda_claim_never_infers_from_nvidia_hardware(monkeypatch) -> None:
    monkeypatch.setattr(
        system_info.bootstrap,
        "detect_hardware",
        lambda: system_info.bootstrap.HardwareProfile(gpu_name="NVIDIA GeForce GTX 1650", gpu_vram_gib=4.0),
    )
    monkeypatch.setattr(
        system_info,
        "_torch_state",
        lambda: {"installed": True, "version": "2.14.0+cpu", "cuda_available": False, "torch_cuda_version": None, "device_count": 0, "devices": []},
    )
    monkeypatch.setattr(system_info, "_nvidia_state", lambda: {"available": True, "gpus": [{"name": "NVIDIA GeForce GTX 1650"}]})
    monkeypatch.setattr(system_info, "_pip_check", lambda: {"passed": True, "exit_code": 0, "output": "No broken requirements found."})

    report = system_info.build_report("default")

    assert report["hardware"]["gpu_name"]
    assert report["torch"]["cuda_available"] is False
    assert report["assessment"]["cuda_claim"] == "not_runtime_verified"
