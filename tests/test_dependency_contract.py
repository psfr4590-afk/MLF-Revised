from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_REQUIREMENTS = ROOT / "requirements.txt"
TORCH_REQUIREMENTS = ROOT / "requirements-torch.txt"


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_core_requirements_keep_pytorch_host_specific():
    core = _requirement_lines(CORE_REQUIREMENTS)
    torch = _requirement_lines(TORCH_REQUIREMENTS)

    assert not any(line.lower().startswith("torch") for line in core)
    assert torch == ["torch>=2.5,<3"]


def test_bootstrap_dependency_files_exist():
    bootstrap = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
    assert 'REQUIREMENTS = ROOT / "requirements.txt"' in bootstrap
    assert 'TORCH_REQUIREMENTS = ROOT / "requirements-torch.txt"' in bootstrap
    assert 'CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"' in bootstrap
