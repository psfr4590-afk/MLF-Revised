from pathlib import Path

import pytest
import yaml

from pipeline.embedder.semantic_dedup import SemanticDeduplicator
from pipeline.types import Document


def test_starter_semantic_dedup_never_downloads_model():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["embed_dedup"]

    assert cfg["mode"] == "auto"
    assert cfg["allow_model_download"] is False
    assert cfg["model_path"] is None


def test_auto_mode_falls_back_when_local_model_is_unavailable(monkeypatch):
    import pipeline.embedder.semantic_dedup as mod

    monkeypatch.setattr(mod, "ST_AVAILABLE", True)
    monkeypatch.setattr(mod, "FAISS_AVAILABLE", True)
    d = SemanticDeduplicator({"mode": "auto", "allow_model_download": False, "similarity_threshold": 0.5})

    def fail_load():
        raise OSError("model not cached")

    monkeypatch.setattr(d, "_load_model", fail_load)
    docs = [
        Document(doc_id="1", url="https://example.com/1", text="alpha beta gamma", final_weight=1.0),
        Document(doc_id="2", url="https://example.com/2", text="alpha beta gamma", final_weight=2.0),
    ]

    assert [d.doc_id for d in d.run(docs)] == ["2"]


def test_embedding_mode_fails_clearly_without_local_model(monkeypatch):
    import pipeline.embedder.semantic_dedup as mod

    monkeypatch.setattr(mod, "ST_AVAILABLE", True)
    monkeypatch.setattr(mod, "FAISS_AVAILABLE", True)
    d = SemanticDeduplicator({"mode": "embedding", "model": "all-MiniLM-L6-v2"})
    monkeypatch.setattr(d, "_load_model", lambda: (_ for _ in ()).throw(OSError("not cached")))

    docs = [Document(doc_id="1", url="https://example.com/1", text="alpha beta gamma", final_weight=1.0)]
    with pytest.raises(RuntimeError, match="locally available model"):
        d.run(docs)


def test_model_download_policy_is_forwarded_to_sentence_transformers(monkeypatch):
    import pipeline.embedder.semantic_dedup as mod

    calls = []

    class FakeModel:
        def __init__(self, model_ref, **kwargs):
            calls.append((model_ref, kwargs))

        def get_sentence_embedding_dimension(self):
            return 384

    monkeypatch.setattr(mod, "ST_AVAILABLE", True)
    monkeypatch.setattr(mod, "SentenceTransformer", FakeModel)

    # Offline is the default policy; explicit opt-in may omit local_files_only.
    SemanticDeduplicator({"model": "test-model", "allow_model_download": False})._load_model()
    SemanticDeduplicator({"model": "test-model", "allow_model_download": True})._load_model()

    assert calls == [
        ("test-model", {"local_files_only": True}),
        ("test-model", {}),
    ]
