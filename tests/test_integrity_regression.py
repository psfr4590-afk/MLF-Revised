from __future__ import annotations

import hashlib
import json

import pytest

from pipeline.integrity import artifact_valid, atomic_jsonl_write, manifest_path, sha256_file, write_manifest


def test_sha256_file_matches_standard_digest(tmp_path):
    path = tmp_path / "payload.bin"
    data = b"model-lab-integrity" * 17
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_manifest_path_is_adjacent_to_artifact(tmp_path):
    path = tmp_path / "artifact.jsonl"
    assert manifest_path(path) == tmp_path / "artifact.jsonl.manifest.json"


def test_write_manifest_and_artifact_valid_roundtrip(tmp_path):
    path = tmp_path / "artifact.jsonl"
    path.write_text('{"id":1}\n', encoding="utf-8")
    provenance = {"stage": "test", "input_sha256": "a" * 64}
    manifest = write_manifest(path, kind="jsonl", rows=1, provenance=provenance, extra={"marker": "ok"})
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["kind"] == "jsonl"
    assert payload["rows"] == 1
    assert payload["provenance"] == provenance
    assert payload["marker"] == "ok"
    assert artifact_valid(path, expected_provenance=provenance)


def test_artifact_valid_rejects_missing_artifact_or_manifest(tmp_path):
    path = tmp_path / "missing.bin"
    assert not artifact_valid(path)
    path.write_bytes(b"data")
    assert not artifact_valid(path)


def test_artifact_valid_rejects_tampered_bytes(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    write_manifest(path, kind="binary")
    path.write_bytes(b"tampered")
    assert not artifact_valid(path)


def test_artifact_valid_rejects_schema_kind_size_and_provenance_mismatch(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    write_manifest(path, kind="binary", provenance={"stage": "one"})
    manifest = manifest_path(path)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = 999
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert not artifact_valid(path)

    write_manifest(path, kind="binary", provenance={"stage": "one"})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["kind"] = ""
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert not artifact_valid(path)

    write_manifest(path, kind="binary", provenance={"stage": "one"})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["size"] += 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert not artifact_valid(path)

    write_manifest(path, kind="binary", provenance={"stage": "one"})
    assert not artifact_valid(path, expected_provenance={"stage": "two"})


def test_artifact_valid_rejects_malformed_manifest(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"data")
    manifest_path(path).write_text("not-json", encoding="utf-8")
    assert not artifact_valid(path)


def test_artifact_valid_validates_optional_source_identity(tmp_path):
    source = tmp_path / "source.txt"
    artifact = tmp_path / "artifact.bin"
    source.write_text("source-v1", encoding="utf-8")
    artifact.write_bytes(b"artifact")
    write_manifest(
        artifact,
        kind="binary",
        extra={"source": str(source), "source_sha256": sha256_file(source)},
    )
    assert artifact_valid(artifact)

    payload = json.loads(manifest_path(artifact).read_text(encoding="utf-8"))
    payload["source_sha256"] = "0" * 64
    manifest_path(artifact).write_text(json.dumps(payload), encoding="utf-8")
    assert not artifact_valid(artifact)


def test_artifact_valid_rejects_one_sided_source_metadata(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    write_manifest(artifact, kind="binary", extra={"source": str(tmp_path / "missing")})
    assert not artifact_valid(artifact)


def test_atomic_jsonl_write_accepts_objects_and_valid_serialized_records(tmp_path):
    path = tmp_path / "records.jsonl"
    count = atomic_jsonl_write(path, [{"id": 1}, '{"id":2,"text":"ok"}'])
    assert count == 2
    assert path.read_text(encoding="utf-8") == '{"id":1}\n{"id":2,"text":"ok"}\n'


def test_atomic_jsonl_write_accepts_callable_producer(tmp_path):
    path = tmp_path / "records.jsonl"
    count = atomic_jsonl_write(path, lambda: ({"id": i} for i in range(2)))
    assert count == 2
    assert [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()] == [0, 1]


def test_atomic_jsonl_write_rejects_invalid_serialized_json_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "records.jsonl"
    with pytest.raises(ValueError, match="pre-serialized JSONL item"):
        atomic_jsonl_write(path, ["not-json"])
    assert not path.exists()
    assert list(tmp_path.glob("records.jsonl.*.tmp")) == []


def test_atomic_jsonl_write_rejects_empty_iterable(tmp_path):
    path = tmp_path / "records.jsonl"
    with pytest.raises(RuntimeError, match="Refusing to commit empty artifact"):
        atomic_jsonl_write(path, [])
    assert not path.exists()
    assert list(tmp_path.glob("records.jsonl.*.tmp")) == []
