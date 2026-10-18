import json

import pytest

from pipeline.integrity import artifact_valid, atomic_jsonl_write, manifest_path, sha256_file, write_manifest


def test_artifact_valid_rejects_changed_provenance(tmp_path):
    path = tmp_path / "artifact.jsonl"
    path.write_text('{"id":1}\n', encoding="utf-8")
    write_manifest(path, kind="clean", provenance={"schema": 1, "input_sha256": "old"})

    assert artifact_valid(path)
    assert not artifact_valid(path, expected_provenance={"schema": 1, "input_sha256": "new"})


def test_artifact_valid_rejects_changed_source(tmp_path):
    source = tmp_path / "source.jsonl"
    artifact = tmp_path / "artifact.jsonl"
    source.write_text("old\n", encoding="utf-8")
    artifact.write_text("derived\n", encoding="utf-8")
    manifest = {
        "schema": 2,
        "kind": "clean",
        "path": str(artifact),
        "size": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
        "source": str(source),
        "source_sha256": sha256_file(source),
    }
    manifest_path(artifact).write_text(json.dumps(manifest), encoding="utf-8")

    assert artifact_valid(artifact)
    source.write_text("changed\n", encoding="utf-8")
    assert not artifact_valid(artifact)


def test_artifact_valid_rejects_incomplete_source_metadata(tmp_path):
    path = tmp_path / "artifact.jsonl"
    source = tmp_path / "source.jsonl"
    path.write_text("data\n", encoding="utf-8")
    source.write_text("source\n", encoding="utf-8")
    manifest = {
        "schema": 2,
        "kind": "clean",
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "source": str(source),
    }
    manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
    assert not artifact_valid(path)


def test_artifact_valid_rejects_unknown_manifest_schema(tmp_path):
    path = tmp_path / "artifact.jsonl"
    path.write_text("data\n", encoding="utf-8")
    manifest = {
        "schema": 999,
        "kind": "clean",
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
    assert not artifact_valid(path)


def test_artifact_valid_rejects_missing_manifest_kind(tmp_path):
    path = tmp_path / "artifact.jsonl"
    path.write_text("data\n", encoding="utf-8")
    manifest = {
        "schema": 2,
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    manifest_path(path).write_text(json.dumps(manifest), encoding="utf-8")
    assert not artifact_valid(path)


def test_artifact_valid_rejects_size_and_hash_mismatch(tmp_path):
    path = tmp_path / "artifact.jsonl"
    path.write_text("data\n", encoding="utf-8")
    write_manifest(path, kind="clean")
    path.write_text("tampered\n", encoding="utf-8")
    assert not artifact_valid(path)


def test_artifact_valid_rejects_missing_or_empty_artifact(tmp_path):
    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert not artifact_valid(missing)
    assert not artifact_valid(empty)


def test_write_manifest_records_rows_and_extra(tmp_path):
    path = tmp_path / "artifact.jsonl"
    path.write_text("one\ntwo\n", encoding="utf-8")
    result = write_manifest(path, kind="clean", rows=2, extra={"marker": "x"})
    data = json.loads(result.read_text(encoding="utf-8"))
    assert result == manifest_path(path)
    assert data["rows"] == 2
    assert data["marker"] == "x"
    assert data["sha256"] == sha256_file(path)


def test_atomic_jsonl_write_commits_nonempty_output(tmp_path):
    path = tmp_path / "artifact.jsonl"
    count = atomic_jsonl_write(path, lambda: iter([{"id": 1}, {"id": 2}]))
    assert count == 2
    assert path.read_text(encoding="utf-8").splitlines() == ['{"id":1}', '{"id":2}']


def test_atomic_jsonl_write_removes_temp_file_on_failure(tmp_path):
    path = tmp_path / "artifact.jsonl"

    def failing_producer():
        yield {"id": 1}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        atomic_jsonl_write(path, failing_producer)
    assert not path.exists()
    assert not list(tmp_path.glob("artifact.jsonl.*.tmp"))


def test_atomic_jsonl_write_rejects_empty_output(tmp_path):
    path = tmp_path / "artifact.jsonl"
    with pytest.raises(RuntimeError, match="empty artifact"):
        atomic_jsonl_write(path, lambda: iter(()))
    assert not path.exists()
