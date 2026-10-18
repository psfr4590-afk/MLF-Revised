from __future__ import annotations

import pytest

from scripts.export_gguf import _enforce_training_provenance


def test_export_refuses_checkpoint_without_training_provenance(tmp_path):
    with pytest.raises(RuntimeError, match="legacy provenance schema"):
        _enforce_training_provenance(tmp_path, {"provenance": {"schema": 1}})


def test_export_refuses_incomplete_provenance(tmp_path):
    with pytest.raises(RuntimeError, match="provenance is incomplete"):
        _enforce_training_provenance(tmp_path, {"provenance": {"schema": 3}})
