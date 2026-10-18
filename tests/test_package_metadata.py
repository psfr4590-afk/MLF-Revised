from pathlib import Path
import json
import tomllib


ROOT = Path(__file__).resolve().parents[1]


_RUNTIME_REQUIREMENTS = {
    "sentence-transformers",
    "faiss-cpu",
    "pyyaml",
    "defusedxml",
    "requests",
    "beautifulsoup4",
    "lxml",
    "numpy",
    "packaging",
    "tokenizers",
    "safetensors",
    "datasets",
    "huggingface_hub",
    "sentencepiece",
    "trafilatura",
    "langdetect",
    "pypdf",
    "openpyxl",
    "python-docx",
    "python-pptx",
    "fastapi",
    "starlette",
    "uvicorn",
    "jinja2",
    "pydantic",
    "httpx",
    "cryptography",
    "pyopenssl",
    "psutil",
}

_VERIFICATION_REQUIREMENTS = {"pytest", "pytest-cov", "ruff"}


def _requirement_name(line: str) -> str:
    return line.split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].split("!", 1)[0].strip().lower()


def test_project_metadata_declares_supported_python_and_cli():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert project["name"] == "model-lab-framework"
    assert project["version"] == "1.3.0rc1"
    assert project["requires-python"] == ">=3.11,<3.15"
    assert project["scripts"]["mlab"] == "run_pipeline:main"


def test_packaging_dependencies_match_runtime_contract():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    declared = {
        _requirement_name(dependency): dependency
        for dependency in metadata["project"]["dependencies"]
    }

    assert [line.strip() for line in requirements.splitlines() if line.strip() and not line.lstrip().startswith("#")] == ["-e ."]
    assert set(declared) == _RUNTIME_REQUIREMENTS
    assert all(not name.startswith("torch") for name in declared)


def test_torch_has_a_separate_host_specific_requirement():
    torch_requirements = (ROOT / "requirements-torch.txt").read_text(encoding="utf-8")
    assert "torch>=2.5,<3" in torch_requirements


def test_build_manifest_version_matches_package_version():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_manifest = json.loads(
        (ROOT / "docs" / "release" / "BUILD_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert build_manifest["version"] == metadata["project"]["version"]
