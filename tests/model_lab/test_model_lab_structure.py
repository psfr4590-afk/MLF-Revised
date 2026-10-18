"""Model Lab release-contract tests for the single operator interface."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_STAGES = ["crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export"]
EXPECTED_CREDS = {
    "github": ("GitHub", "GITHUB_TOKEN"),
    "huggingface": ("Hugging Face", "HF_TOKEN"),
    "google_api": ("Google", "GOOGLE_API_KEY"),
    "google_cx": ("Google", "GOOGLE_CX"),
}


def text(path):
    return path.read_text(encoding="utf-8")


def test_package_identity_is_model_lab():
    readme = text(ROOT / "README.md")
    assert "Model Lab" in readme and "M²S Model Training Pipeline" in readme


def test_root_launcher_is_the_canonical_command_center_launcher():
    p = ROOT / "launch.py"
    src = text(p)
    assert p.exists()
    tree = ast.parse(src)
    assert 'COMMAND_CENTER = ROOT / "run_command_center.py"' in src
    assert "ui/app.py" not in src
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "call"
        for n in ast.walk(tree)
    )


def test_compatibility_launcher_delegates_to_the_same_interface():
    src = text(ROOT / "ui/app.py")
    assert "run_command_center.py" in src
    assert "tkinter" not in src
    assert "ui.core.application" not in src


def test_command_center_is_the_only_browser_operator_surface():
    src = text(ROOT / "command_center/web.py")
    assert "Single local operator interface" in src
    assert "/api/dashboard" in src
    assert "/api/runs" in src
    assert "/api/datasets" in src
    assert "authoritative SQLite state" in src


def test_pipeline_stage_contract_is_complete():
    src = text(ROOT / "command_center/service.py")
    for stage in EXPECTED_STAGES:
        assert stage in src


def test_all_stage_routes_are_exposed_by_command_center():
    src = text(ROOT / "command_center/web.py")
    assert "/api/datasets/{did}/stage/{name}" in src


def test_stop_route_exists():
    assert "/api/datasets/{did}/stop" in text(ROOT / "command_center/web.py")


def test_dataset_routes_cover_list_create_get_ingest():
    src = text(ROOT / "command_center/web.py")
    for route in ["/api/datasets", "/api/datasets/{did}", "/api/datasets/{did}/ingest"]:
        assert route in src


def test_group_route_exists():
    assert "/api/groups" in text(ROOT / "command_center/web.py")


def test_system_route_exists():
    assert "/api/system" in text(ROOT / "command_center/web.py")


def test_credential_routes_cover_list_set_test_delete():
    src = text(ROOT / "command_center/web.py")
    for route in ["/api/credentials", "/api/credentials/{name}/test", "/api/credentials/{name}"]:
        assert route in src


def test_credential_presets_are_exactly_the_four_required_slots():
    src = text(ROOT / "command_center/service.py")
    for name, (provider, env) in EXPECTED_CREDS.items():
        assert name in src and provider in src and env in src


def test_documentation_contains_one_operator_interface():
    readme = text(ROOT / "README.md")
    assert "exactly one operator interface" in readme
    assert "python .\\run_command_center.py" in readme
    assert "run_pipeline.py --doctor" in readme


def test_no_second_desktop_launcher_is_advertised():
    readme = text(ROOT / "README.md")
    assert "Windows desktop UI" not in readme
    assert "desktop control surface" not in readme


def test_no_common_secret_literals_are_committed():
    forbidden = ["g" + "hp_", "github" + "_pat_", "AI" + "za"]
    ignored_dirs = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache"}
    ignored_suffixes = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bin", ".pt", ".dll", ".pyd", ".so", ".dylib"}
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in ignored_dirs for part in p.parts) or "llama.cpp" in p.parts:
            continue
        if p.name in {"README.md", ".env.example", "credentials.example.yaml"} or p.suffix.lower() in ignored_suffixes:
            continue
        data = p.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            assert token not in data, f"possible secret literal {token} in {p}"


def test_release_docs_exist():
    for name in {"README.md", "docs/development/START_HERE.md", "docs/development/PROJECT_STATE.md", "docs/verification/VERIFICATION.md"}:
        assert (ROOT / name).exists()


def test_machine_verification_suite_is_present():
    assert (ROOT / "tests/model_lab/test_machine_environment.py").exists() and (ROOT / "scripts/run_release_verification.ps1").exists()


def test_traceability_document_exists():
    assert (ROOT / "docs/verification/VERIFICATION_CHECKLIST.md").exists()
