"""Test ShardDataLoader SHA256 integrity verification."""
from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pipeline.shardwriter.shard_writer import ShardDataLoader


def test_shard_data_loader_rejects_corrupted_file():
    """Verify that ShardDataLoader detects SHA256 mismatches and raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir)

        # Create a valid shard file with tokens
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
        shard_path = shard_dir / "shard_00000_train.bin"
        tokens.tofile(shard_path)

        # Compute correct SHA256
        digest = ShardDataLoader._compute_sha256(shard_path)

        # Create manifest with correct SHA256
        manifest = {
            "version": 1,
            "dtype": "uint16",
            "sequence_length": 1024,
            "vocab_size": 32000,
            "files": [
                {
                    "name": "shard_00000_train.bin",
                    "size": shard_path.stat().st_size,
                    "sha256": digest,
                }
            ],
        }
        manifest_path = shard_dir / "shards.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Verify loader succeeds with valid manifest and file
        loader = ShardDataLoader(shard_dir, "train", seq_len=2, dtype=np.uint16)
        assert loader is not None

        # Release the Windows memory-mapped handle before mutating/cleaning up.
        del loader
        gc.collect()

        # Now corrupt the shard file
        with shard_path.open("r+b") as f:
            f.seek(2)
            f.write(b"\xFF\xFF")  # Corrupt 2 bytes

        # Create a new loader instance; it should detect the corruption
        with pytest.raises(RuntimeError, match="Shard integrity check failed"):
            ShardDataLoader(shard_dir, "train", seq_len=2, dtype=np.uint16)


def test_shard_data_loader_rejects_missing_sha256():
    """Verify that loader rejects manifest entries without SHA256 field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir)

        # Create a shard file
        tokens = np.array([1, 2, 3, 4], dtype=np.uint16)
        shard_path = shard_dir / "shard_00000_train.bin"
        tokens.tofile(shard_path)

        # Create manifest WITHOUT sha256 field
        manifest = {
            "version": 1,
            "dtype": "uint16",
            "sequence_length": 1024,
            "vocab_size": 32000,
            "files": [
                {
                    "name": "shard_00000_train.bin",
                    "size": shard_path.stat().st_size,
                    # Missing: "sha256": "..."
                }
            ],
        }
        manifest_path = shard_dir / "shards.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Loader should reject this manifest
        with pytest.raises(RuntimeError, match="missing SHA256"):
            ShardDataLoader(shard_dir, "train", seq_len=2, dtype=np.uint16)


def test_shard_data_loader_rejects_size_mismatch():
    """Verify that loader rejects files with size mismatches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir)

        # Create a shard file
        tokens = np.array([1, 2, 3, 4], dtype=np.uint16)
        shard_path = shard_dir / "shard_00000_train.bin"
        tokens.tofile(shard_path)

        # Create manifest with wrong size
        manifest = {
            "version": 1,
            "dtype": "uint16",
            "sequence_length": 1024,
            "vocab_size": 32000,
            "files": [
                {
                    "name": "shard_00000_train.bin",
                    "size": 9999,  # Wrong size
                    "sha256": "dummy_sha",
                }
            ],
        }
        manifest_path = shard_dir / "shards.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Loader should reject this
        with pytest.raises(RuntimeError, match="Shard manifest mismatch"):
            ShardDataLoader(shard_dir, "train", seq_len=2, dtype=np.uint16)
