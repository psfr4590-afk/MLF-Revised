import numpy as np

from pipeline.embedder.semantic_dedup import SemanticDeduplicator
from pipeline.types import Document


def doc(i, text, weight):
    return Document(doc_id=str(i), url=f"https://example.com/{i}", text=text, final_weight=weight)


def test_fallback_stream_dedup_crosses_buffer_boundary(monkeypatch):
    import pipeline.embedder.semantic_dedup as mod
    monkeypatch.setattr(mod, "ST_AVAILABLE", False)
    d = SemanticDeduplicator({"mode": "fallback", "similarity_threshold": 0.5})
    docs = iter([
        doc(1, "alpha beta gamma delta", 1.0),
        doc(2, "totally different corpus material", 1.0),
        doc(3, "alpha beta gamma delta", 2.0),
    ])
    out = list(d.stream(docs, buffer_size=1))
    assert [x.doc_id for x in out] == ["3", "2"]


def test_embedding_stream_updates_representative_vector_when_weight_wins(monkeypatch):
    import pipeline.embedder.semantic_dedup as mod
    monkeypatch.setattr(mod, "ST_AVAILABLE", True)
    monkeypatch.setattr(mod, "FAISS_AVAILABLE", True)

    vectors = {
        "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([0.8, 0.6, 0.0], dtype=np.float32),
        "c": np.array([0.30901699, 0.95105652, 0.0], dtype=np.float32),
    }
    d = SemanticDeduplicator({"similarity_threshold": 0.8})
    monkeypatch.setattr(d, "_load_model", lambda: None)
    monkeypatch.setattr(d, "_embed", lambda texts: np.asarray([vectors[t] for t in texts], dtype=np.float32))

    out = list(d.stream(iter([
        doc(1, "a", 1.0),
        doc(2, "b", 2.0),
        doc(3, "c", 1.5),
    ]), buffer_size=1))

    # b replaces a as the representative. c is similar to b but not to the
    # original a, so c must be dropped only if the replacement vector is kept.
    assert [x.doc_id for x in out] == ["2"]
