from __future__ import annotations
import pytest

import json

from scripts import verify_release


def _seed_report_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_release, "ROOT", tmp_path)
    monkeypatch.setattr(
        verify_release,
        "_git_state",
        lambda _root: {"commit": "abc123", "branch": "main", "dirty": False, "status_entries": []},
    )
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "docs" / "development").mkdir(parents=True)
    (tmp_path / "docs" / "development" / "START_HERE.md").write_text("start", encoding="utf-8")


def test_rc_report_is_single_machine_readable_gate(tmp_path, monkeypatch):
    _seed_report_environment(tmp_path, monkeypatch)
    verify_release.NATIVE_EVIDENCE.clear()
    verify_release.NATIVE_EVIDENCE.update({
        "provenance": True, "lineage": True, "dataset": True, "source_retrieval": True,
        "checkpoint": True, "best_checkpoint": True, "reproducibility": True,
        "hardware": True, "auto_sizing": True, "eta": True, "evaluation": True,
        "sqlite": True, "export": True, "inference": True,
        "ui_backend_connectivity": True, "clean_clone": True, "secrets": True,
        "package": True, "package_detail": "valid wheel fixture",
    })

    assert verify_release._write_release_report(static_failures=[], native_requested=True, native_rc=0) == 0
    report = json.loads((tmp_path / "release-evidence" / "release_report.json").read_text(encoding="utf-8"))
    assert report["release"] == "ML-FRAMEWORK RC"
    assert report["status"] == "PASS"
    assert {item["name"] for item in report["checks"]} >= {
        "tests", "configs", "provenance", "lineage", "dataset", "source_retrieval",
        "checkpoint", "best_checkpoint", "reproducibility", "hardware", "auto_sizing",
        "eta", "evaluation", "sqlite", "ui_backend_connectivity", "export", "inference",
        "clean_clone", "documentation", "secrets", "git_state", "package", "release_artifacts",
    }


def test_rc_report_fails_on_unknown_required_evidence(tmp_path, monkeypatch):
    _seed_report_environment(tmp_path, monkeypatch)
    verify_release.NATIVE_EVIDENCE.clear()
    for name in (
        "provenance", "lineage", "dataset", "source_retrieval", "checkpoint",
        "best_checkpoint", "reproducibility", "hardware", "auto_sizing", "eta",
        "evaluation", "sqlite", "export", "inference", "secrets",
    ):
        verify_release.NATIVE_EVIDENCE[name] = True

    assert verify_release._write_release_report(static_failures=[], native_requested=True, native_rc=0) == 2
    report = json.loads((tmp_path / "release-evidence" / "release_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["ui_backend_connectivity"] == "UNKNOWN"
    assert statuses["clean_clone"] == "UNKNOWN"
    assert statuses["package"] == "UNKNOWN"


def test_rc_report_fails_when_package_validation_fails(tmp_path, monkeypatch):
    _seed_report_environment(tmp_path, monkeypatch)
    verify_release.NATIVE_EVIDENCE.clear()
    verify_release.NATIVE_EVIDENCE.update({
        "provenance": True, "lineage": True, "dataset": True, "source_retrieval": True,
        "checkpoint": True, "best_checkpoint": True, "reproducibility": True,
        "hardware": True, "auto_sizing": True, "eta": True, "evaluation": True,
        "sqlite": True, "export": True, "inference": True,
        "ui_backend_connectivity": True, "clean_clone": True, "secrets": True,
        "package": False, "package_detail": "wheel missing a required module",
    })
    assert verify_release._write_release_report(
        static_failures=[], native_requested=True, native_rc=0, profile="rc"
    ) == 2
    report = json.loads(
        (tmp_path / "release-evidence" / "release_report.json").read_text(encoding="utf-8")
    )
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert report["status"] == "FAIL"
    assert statuses["package"] == "FAIL"



def test_source_retrieval_smoke_exercises_offline_upstream_stages():
    assert verify_release._source_retrieval_smoke() is True


def test_offline_pipeline_smoke_does_not_claim_live_source_retrieval(monkeypatch):
    verify_release.NATIVE_EVIDENCE.clear()
    monkeypatch.setattr(verify_release, "_source_retrieval_smoke", lambda: True)

    assert verify_release._record_source_pipeline_integration_smoke() is True
    assert verify_release.NATIVE_EVIDENCE["source_pipeline_integration"] is True
    assert "source_retrieval" not in verify_release.NATIVE_EVIDENCE


def test_rc_report_fails_when_source_retrieval_evidence_is_missing(tmp_path, monkeypatch):
    _seed_report_environment(tmp_path, monkeypatch)
    verify_release.NATIVE_EVIDENCE.clear()
    verify_release.NATIVE_EVIDENCE.update({
        "provenance": True, "lineage": True, "dataset": True,
        "source_retrieval": None, "checkpoint": True, "best_checkpoint": True,
        "reproducibility": True, "hardware": True, "auto_sizing": False,
        "eta": True, "evaluation": True, "sqlite": True, "export": True,
        "inference": True, "ui_backend_connectivity": True, "clean_clone": True,
        "secrets": True, "package": True, "package_detail": "valid wheel fixture",
    })

    assert verify_release._write_release_report(
        static_failures=[], native_requested=True, native_rc=0, profile="rc"
    ) == 2
    report = json.loads(
        (tmp_path / "release-evidence" / "release_report.json").read_text(encoding="utf-8")
    )
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert report["status"] == "FAIL"
    assert statuses["auto_sizing"] == "PASS"
    assert statuses["source_retrieval"] == "UNKNOWN"
    assert "source_retrieval" in report["required_checks"]


def test_rc_source_retrieval_evidence_must_be_explicitly_true(tmp_path, monkeypatch):
    _seed_report_environment(tmp_path, monkeypatch)
    verify_release.NATIVE_EVIDENCE.clear()
    verify_release.NATIVE_EVIDENCE.update({
        "provenance": True, "lineage": True, "dataset": True,
        "source_retrieval": False, "checkpoint": True, "best_checkpoint": True,
        "reproducibility": True, "hardware": True, "auto_sizing": False,
        "eta": True, "evaluation": True, "sqlite": True, "export": True,
        "inference": True, "ui_backend_connectivity": True, "clean_clone": True,
        "secrets": True, "package": True, "package_detail": "valid wheel fixture",
    })

    assert verify_release._write_release_report(
        static_failures=[], native_requested=True, native_rc=0, profile="rc"
    ) == 2
    report_path = tmp_path / "release-evidence" / "release_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "source_retrieval" in report["required_checks"]
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["source_retrieval"] == "FAIL"

    verify_release.NATIVE_EVIDENCE["source_retrieval"] = True
    assert verify_release._write_release_report(
        static_failures=[], native_requested=True, native_rc=0, profile="rc"
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "source_retrieval" in report["required_checks"]
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["source_retrieval"] == "PASS"


def _mock_wheel_build(tmp_path, monkeypatch, fault=None):
    import zipfile
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.setattr(verify_release, "ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "model-lab-framework"\nversion = "1.3.0rc1"\n',
        encoding="utf-8",
    )

    def fake_run(command, cwd, capture_output, text, check):
        wheel_dir = Path(command[command.index("--wheel-dir") + 1])
        wheel_dir.mkdir(parents=True, exist_ok=True)
        wheel = wheel_dir / "model_lab_framework-1.3.0rc1-py3-none-any.whl"
        if fault == "corrupt":
            wheel.write_bytes(b"not a zip archive")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        version = "9.9.9" if fault == "wrong_version" else "1.3.0rc1"
        modules = [
            "run_pipeline.py", "mlframework.py", "run_command_center.py", "launch.py",
            "scripts/__init__.py", "scripts/export_gguf.py", "scripts/export_cards.py",
        ]
        config_assets = [
            "config/__init__.py",
            "config/pipeline_config.yaml",
            "config/pipeline_config.dataset.yaml",
            "config/pipeline_config.full.yaml",
            "config/pipeline_config.smoke.yaml",
            "config/dataset_groups.yaml",
            "config/dataset_groups.smoke.yaml",
            "config/dataset_profiles.yaml",
            "config/dataset_source_policy.yaml",
            "config/source_weights.yaml",
            "config/cleaner_config.yaml",
            "config/seed_urls.txt",
        ]
        if fault == "missing_config":
            config_assets.remove("config/dataset_groups.yaml")
        if fault == "missing_export_gguf":
            modules.remove("scripts/export_gguf.py")
        if fault == "missing_export_cards":
            modules.remove("scripts/export_cards.py")
        if fault == "missing_module":
            modules.remove("launch.py")

        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "model_lab_framework-1.3.0rc1.dist-info/METADATA",
                f"Metadata-Version: 2.1\nName: model-lab-framework\nVersion: {version}\n\n",
            )
            archive.writestr(
                "model_lab_framework-1.3.0rc1.dist-info/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            archive.writestr("model_lab_framework-1.3.0rc1.dist-info/RECORD", "")
            archive.writestr(
                "model_lab_framework-1.3.0rc1.dist-info/entry_points.txt",
                "[console_scripts]\n"
                "mlab = run_pipeline:main\n"
                "mlframework = mlframework:main\n"
                "mlab-command-center = run_command_center:main\n"
                "mlab-launch = launch:main\n",
            )
            for module in modules:
                archive.writestr(module, "# test module\n")
            for asset in config_assets:
                archive.writestr(asset, "# test config\n")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(verify_release.subprocess, "run", fake_run)


def test_build_and_validate_wheel_accepts_valid_wheel(tmp_path, monkeypatch):
    _mock_wheel_build(tmp_path, monkeypatch)
    ok, detail = verify_release._build_and_validate_wheel()
    assert ok is True
    assert "Validated model_lab_framework-1.3.0rc1-py3-none-any.whl" in detail


def test_build_and_validate_wheel_rejects_missing_config_asset(tmp_path, monkeypatch):
    _mock_wheel_build(tmp_path, monkeypatch, fault="missing_config")
    ok, detail = verify_release._build_and_validate_wheel()
    assert ok is False
    assert "Missing wheel configuration assets" in detail
    assert "config/dataset_groups.yaml" in detail


def test_build_and_validate_wheel_rejects_missing_gguf_exporter(tmp_path, monkeypatch):
    _mock_wheel_build(tmp_path, monkeypatch, fault="missing_export_gguf")
    ok, detail = verify_release._build_and_validate_wheel()
    assert ok is False
    assert "scripts/export_gguf.py" in detail


def test_build_and_validate_wheel_rejects_missing_export_card_helper(tmp_path, monkeypatch):
    _mock_wheel_build(tmp_path, monkeypatch, fault="missing_export_cards")
    ok, detail = verify_release._build_and_validate_wheel()
    assert ok is False
    assert "scripts/export_cards.py" in detail


def test_build_and_validate_wheel_rejects_missing_module(tmp_path, monkeypatch):
    _mock_wheel_build(tmp_path, monkeypatch, fault="missing_module")
    ok, detail = verify_release._build_and_validate_wheel()
    assert ok is False
    assert "Missing wheel modules" in detail


def test_build_and_validate_wheel_rejects_wrong_version(tmp_path, monkeypatch):
    _mock_wheel_build(tmp_path, monkeypatch, fault="wrong_version")
    ok, detail = verify_release._build_and_validate_wheel()
    assert ok is False
    assert "version does not match" in detail


def test_build_and_validate_wheel_rejects_corrupt_archive(tmp_path, monkeypatch):
    _mock_wheel_build(tmp_path, monkeypatch, fault="corrupt")
    ok, detail = verify_release._build_and_validate_wheel()
    assert ok is False
    assert "Wheel validation failed" in detail

def test_auto_sizing_evidence_reads_effective_training_configuration():
    evidence = verify_release._auto_sizing_evidence({
        "training_metadata": {
            "effective_configuration": {"auto_size": True},
        },
    })
    assert evidence is True


def test_auto_sizing_evidence_preserves_explicit_disabled_value():
    evidence = verify_release._auto_sizing_evidence({
        "training_metadata": {
            "effective_configuration": {"auto_size": False},
        },
    })
    assert evidence is False


def test_auto_sizing_evidence_supports_legacy_and_unknown_values():
    assert verify_release._auto_sizing_evidence({"auto_size": True}) is True
    assert verify_release._auto_sizing_evidence({"auto_size": False}) is False
    assert verify_release._auto_sizing_evidence({}) is None
    assert verify_release._auto_sizing_evidence(None) is None

def test_auto_sizing_evidence_reads_actual_run_manifest_shape():
    evidence = verify_release._auto_sizing_evidence({
        "effective_configuration": {"auto_size": False},
        "requested_configuration": {"auto_size": False},
    })
    assert evidence is False


def test_best_checkpoint_requires_real_best_manifest_contract():
    verify_release._validate_best_checkpoint_metadata({
        "checkpoint_kind": "best",
        "step": 12,
        "best_step": 12,
        "best_val_loss": 1.25,
    })


@pytest.mark.parametrize("metadata", [
    {"checkpoint_kind": "periodic", "step": 12, "best_step": 12, "best_val_loss": 1.25},
    {"checkpoint_kind": "best", "step": 12, "best_step": None, "best_val_loss": 1.25},
    {"checkpoint_kind": "best", "step": 12, "best_step": -1, "best_val_loss": 1.25},
    {"checkpoint_kind": "best", "step": 12, "best_step": 12, "best_val_loss": float("nan")},
    {"checkpoint_kind": "best", "step": 12, "best_step": 12, "best_val_loss": float("inf")},
    {"checkpoint_kind": "best", "step": True, "best_step": 12, "best_val_loss": 1.25},
])
def test_best_checkpoint_rejects_invalid_manifest(metadata):
    with pytest.raises(RuntimeError):
        verify_release._validate_best_checkpoint_metadata(metadata)


def _valid_identity_fixture():
    return {
        "dataset_config_sha256": "a" * 64,
        "tokenizer_config_sha256": "b" * 64,
        "shard_config_sha256": "c" * 64,
        "export_config_sha256": "d" * 64,
        "source_definition_sha256": "e" * 64,
    }


def test_checkpoint_identity_chain_requires_and_matches_all_stable_hashes():
    expected = _valid_identity_fixture()
    actual = dict(expected)
    # The legacy pipeline hash is deliberately not part of strict equality.
    expected["pipeline_config_sha256"] = "1" * 64
    actual["pipeline_config_sha256"] = "2" * 64
    verify_release._validate_checkpoint_identity_chain(expected, actual)


@pytest.mark.parametrize("missing_key", [
    "dataset_config_sha256",
    "tokenizer_config_sha256",
    "shard_config_sha256",
    "export_config_sha256",
    "source_definition_sha256",
])
def test_checkpoint_identity_chain_rejects_missing_hash(missing_key):
    expected = _valid_identity_fixture()
    actual = dict(expected)
    del actual[missing_key]
    with pytest.raises(RuntimeError, match=missing_key):
        verify_release._validate_checkpoint_identity_chain(expected, actual)


def test_checkpoint_identity_chain_rejects_mismatch_and_malformed_hash():
    expected = _valid_identity_fixture()
    actual = dict(expected)
    actual["shard_config_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="identity mismatch"):
        verify_release._validate_checkpoint_identity_chain(expected, actual)

    actual = dict(expected)
    actual["export_config_sha256"] = "not-a-sha256"
    with pytest.raises(RuntimeError, match="export_config_sha256"):
        verify_release._validate_checkpoint_identity_chain(expected, actual)


@pytest.mark.parametrize("value", [0, 0.0, 1.25, 1e-12])
def test_finite_metric_accepts_numeric_values(value):
    assert verify_release._is_finite_number(value) is True


@pytest.mark.parametrize(
    "value",
    [None, True, False, "1.25", [], {}, float("nan"), float("inf"), float("-inf")],
)
def test_finite_metric_rejects_invalid_values(value):
    assert verify_release._is_finite_number(value) is False


@pytest.mark.parametrize("digest", ["a" * 64, "ABCDEF0123456789" * 4])
def test_sha256_validator_accepts_hex_digest(digest):
    assert verify_release._is_sha256(digest) is True


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "a" * 65, "g" * 64, 123])
def test_sha256_validator_rejects_malformed_digest(digest):
    assert verify_release._is_sha256(digest) is False


def test_dataset_evidence_requires_structured_nonempty_payloads():
    assert verify_release._dataset_evidence_is_valid(
        {"documents": 3}, {"dataset_id": "fixture"}, {"train": {"val_loss": 1.2}}
    ) is True


@pytest.mark.parametrize(
    "payload,identity,metrics",
    [
        (None, {"dataset_id": "fixture"}, {"train": {}}),
        ([], {"dataset_id": "fixture"}, {"train": {}}),
        ({}, {"dataset_id": "fixture"}, {"train": {}}),
        ({"documents": 3}, None, {"train": {}}),
        ({"documents": 3}, {}, {"train": {}}),
        ({"documents": 3}, {"dataset_id": "fixture"}, None),
        ({"documents": 3}, {"dataset_id": "fixture"}, {}),
    ],
)
def test_dataset_evidence_rejects_missing_or_unstructured_payloads(payload, identity, metrics):
    assert verify_release._dataset_evidence_is_valid(payload, identity, metrics) is False


def test_evaluation_evidence_requires_finite_validation_losses():
    assert verify_release._evaluation_evidence_is_valid(
        {"val_loss": 2.0, "best_val_loss": 1.5}
    ) is True


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        {},
        {"val_loss": None, "best_val_loss": 1.5},
        {"val_loss": float("nan"), "best_val_loss": 1.5},
        {"val_loss": 2.0, "best_val_loss": float("inf")},
        {"val_loss": "2.0", "best_val_loss": 1.5},
        {"val_loss": True, "best_val_loss": 1.5},
    ],
)
def test_evaluation_evidence_rejects_invalid_validation_losses(metrics):
    assert verify_release._evaluation_evidence_is_valid(metrics) is False


def test_reproducibility_evidence_requires_integer_seed_and_sha256():
    assert verify_release._reproducibility_evidence_is_valid(
        {"seed": 42, "random_state_sha256": "a" * 64}
    ) is True


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        {},
        {"seed": None, "random_state_sha256": "a" * 64},
        {"seed": True, "random_state_sha256": "a" * 64},
        {"seed": "42", "random_state_sha256": "a" * 64},
        {"seed": 42, "random_state_sha256": "not-a-digest"},
        {"seed": 42, "random_state_sha256": "a" * 63},
    ],
)
def test_reproducibility_evidence_rejects_invalid_seed_or_digest(metrics):
    assert verify_release._reproducibility_evidence_is_valid(metrics) is False


def test_run_git_identity_requires_manifest_database_and_current_commit_to_match():
    manifest = {"git": {"commit": "a" * 40}}
    verify_release._validate_run_git_identity(
        manifest=manifest, database_git_sha="a" * 40, expected_git_sha="a" * 40
    )


@pytest.mark.parametrize(
    "manifest,database_git_sha,expected_git_sha",
    [
        ({"git": {"commit": "a" * 40}}, "a" * 40, "b" * 40),
        ({"git": {"commit": "a" * 40}}, "b" * 40, "a" * 40),
        ({}, "a" * 40, "a" * 40),
        ({"git": {"commit": "a" * 40}}, None, "a" * 40),
        ({"git": {"commit": "a" * 40}}, "a" * 40, None),
    ],
)
def test_run_git_identity_rejects_missing_or_mismatched_identity(
    manifest, database_git_sha, expected_git_sha
):
    with pytest.raises(RuntimeError):
        verify_release._validate_run_git_identity(
            manifest=manifest,
            database_git_sha=database_git_sha,
            expected_git_sha=expected_git_sha,
        )


@pytest.mark.parametrize("value", [0, 1, 42])
def test_nonnegative_checkpoint_step_accepts_integer_values(value):
    assert verify_release._is_nonnegative_int(value) is True


@pytest.mark.parametrize("value", [None, -1, True, False, 1.0, "1"])
def test_nonnegative_checkpoint_step_rejects_invalid_values(value):
    assert verify_release._is_nonnegative_int(value) is False


def test_artifact_id_validator_accepts_kind_and_sha256():
    assert verify_release._is_artifact_id("dataset-manifest:" + "a" * 64) is True


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        ": " + "a" * 64,
        "dataset-manifest",
        "dataset-manifest:" + "a" * 63,
        "dataset-manifest:" + "g" * 64,
        "dataset-manifest:" + "a" * 64 + ":extra",
    ],
)
def test_artifact_id_validator_rejects_malformed_ids(value):
    assert verify_release._is_artifact_id(value) is False


def test_lineage_evidence_requires_valid_parent_dataset_and_tokenizer_ids():
    provenance = {
        "run_id": "run-1",
        "parent_artifact_ids": ["shard-manifest:" + "a" * 64],
        "dataset_artifact_id": "dataset-manifest:" + "b" * 64,
        "tokenizer_artifact_id": "tokenizer-manifest:" + "c" * 64,
    }
    assert verify_release._lineage_evidence_is_valid(provenance, "run-1") is True


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "different-run"},
        {"parent_artifact_ids": []},
        {"parent_artifact_ids": ["not-an-artifact-id"]},
        {"dataset_artifact_id": None},
        {"dataset_artifact_id": "dataset-manifest:bad"},
        {"tokenizer_artifact_id": ""},
    ],
)
def test_lineage_evidence_rejects_missing_or_malformed_links(changes):
    provenance = {
        "run_id": "run-1",
        "parent_artifact_ids": ["shard-manifest:" + "a" * 64],
        "dataset_artifact_id": "dataset-manifest:" + "b" * 64,
        "tokenizer_artifact_id": "tokenizer-manifest:" + "c" * 64,
    }
    provenance.update(changes)
    assert verify_release._lineage_evidence_is_valid(provenance, "run-1") is False


def test_final_checkpoint_requires_final_kind_and_valid_step():
    verify_release._validate_final_checkpoint_metadata(
        {"checkpoint_kind": "final", "step": 12}
    )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"checkpoint_kind": "best", "step": 12},
        {"checkpoint_kind": "final", "step": None},
        {"checkpoint_kind": "final", "step": -1},
        {"checkpoint_kind": "final", "step": True},
        {"checkpoint_kind": "final", "step": 1.0},
    ],
)
def test_final_checkpoint_rejects_invalid_metadata(metadata):
    with pytest.raises(RuntimeError):
        verify_release._validate_final_checkpoint_metadata(metadata)



def test_online_source_evidence_uses_matching_independent_run_manifests(
    tmp_path, monkeypatch
):
    source_path, run_path = _write_live_source_evidence_fixture(tmp_path, monkeypatch)
    assert verify_release._live_source_retrieval_evidence_is_valid(
        source_manifest_path=source_path,
        source_run_manifest_path=run_path,
    ) is True


@pytest.mark.parametrize("run_manifest_changes", [
    {"final_status": "FAIL"},
    {"errors": ["crawler failed"]},
    {"degraded_stages": ["crawl"]},
    {"run_id": "different-run"},
    {"stages": {"crawl": {"status": "SKIPPED"}}},
])
def test_online_source_evidence_rejects_failed_or_mismatched_run(
    tmp_path, monkeypatch, run_manifest_changes
):
    source_path, run_path = _write_live_source_evidence_fixture(tmp_path, monkeypatch)
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    run_manifest.update(run_manifest_changes)
    run_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    assert verify_release._live_source_retrieval_evidence_is_valid(
        source_manifest_path=source_path,
        source_run_manifest_path=run_path,
    ) is False


def test_local_filesystem_retrieval_does_not_claim_online_retrieval(
    tmp_path, monkeypatch
):
    source_path, run_path = _write_live_source_evidence_fixture(tmp_path, monkeypatch)
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_manifest["sources"] = [
        {"kind": "local_files", "identifier": "E:/USB/corpus"}
    ]
    source_manifest["retrieval"] = [{
        "kind": "local_files", "source_mode": "local", "status": "SUCCESS",
        "stats": {"document_count": 12, "request_count": 12},
    }]
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    assert verify_release._live_source_retrieval_evidence_is_valid(
        source_manifest_path=source_path,
        source_run_manifest_path=run_path,
    ) is False


def _write_live_source_evidence_fixture(tmp_path, monkeypatch, *, fault=None):
    monkeypatch.setattr(verify_release, "ROOT", tmp_path)
    run_id = "online-run-1"
    crawl_path = tmp_path / "output" / "runs" / run_id / "01_crawled.jsonl"
    crawl_path.parent.mkdir(parents=True)
    document = {
        "doc_id": "doc-1",
        "source": "web",
        "url": "https://example.com/research",
        "text": "A real fetched document body with enough content to prove non-empty retrieval.",
    }
    if fault == "local_url":
        document["url"] = "file:///tmp/fixture.txt"
    if fault == "empty_body":
        document["text"] = "  "
    crawl_path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    digest = verify_release._sha256(crawl_path)
    if fault == "tampered_artifact":
        digest = "0" * 64

    source_manifest = {
        "schema": 3,
        "run_id": run_id if fault != "mismatched_run_id" else "other-run",
        "source_definition_sha256": "a" * 64,
        "retrieval_started_at": "2026-01-01T00:00:01+00:00",
        "retrieval_completed_at": "2026-01-01T00:00:09+00:00",
        "sources": [{"kind": "web", "identifier": "https://example.com"}],
        "retrieval": [{
            "kind": "web",
            "status": "SUCCESS",
            "stats": {
                "document_count": True if fault == "boolean_count" else 1,
                "request_count": 1,
            },
        }],
    }
    source_path = tmp_path / "source_manifest.json"
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    crawl_stage = {
        "status": "PASS",
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-01-01T00:00:10+00:00",
        "errors": [],
        "outputs": [{"path": crawl_path.relative_to(tmp_path).as_posix(), "sha256": digest}],
    }
    if fault == "failed_crawl":
        crawl_stage["status"] = "FAILED"
    if fault == "timestamp_outside_run":
        source_manifest["retrieval_completed_at"] = "2026-01-01T00:00:11+00:00"
        source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    if fault == "missing_output":
        crawl_stage["outputs"] = []

    run_manifest = {
        "run_id": run_id,
        "final_status": "PASS",
        "errors": [],
        "degraded_stages": [],
        "stages": {"crawl": crawl_stage},
    }
    run_path = tmp_path / "run_manifest.json"
    run_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    return source_path, run_path


def test_live_source_retrieval_requires_hashed_http_document_and_matching_run(
    tmp_path, monkeypatch
):
    source_path, run_path = _write_live_source_evidence_fixture(tmp_path, monkeypatch)
    assert verify_release._live_source_retrieval_evidence_is_valid(
        source_manifest_path=source_path,
        source_run_manifest_path=run_path,
    ) is True


@pytest.mark.parametrize(
    "fault",
    [
        "tampered_artifact",
        "local_url",
        "empty_body",
        "mismatched_run_id",
        "failed_crawl",
        "timestamp_outside_run",
        "missing_output",
        "boolean_count",
    ],
)
def test_live_source_retrieval_rejects_unsubstantiated_or_invalid_evidence(
    tmp_path, monkeypatch, fault
):
    source_path, run_path = _write_live_source_evidence_fixture(
        tmp_path, monkeypatch, fault=fault
    )
    assert verify_release._live_source_retrieval_evidence_is_valid(
        source_manifest_path=source_path,
        source_run_manifest_path=run_path,
    ) is False


def test_live_source_retrieval_is_unknown_without_independent_run_manifest(tmp_path):
    source_path = tmp_path / "source_manifest.json"
    source_path.write_text("{}", encoding="utf-8")
    assert verify_release._live_source_retrieval_evidence_is_valid(
        source_manifest_path=source_path,
    ) is None
