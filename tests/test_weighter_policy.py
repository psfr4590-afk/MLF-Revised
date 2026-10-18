from pipeline.types import Document
from pipeline.weighter.weighter import DomainWeighter


def _doc(doc_id: str, weight: float) -> Document:
    return Document(
        doc_id=doc_id,
        text=f"document {doc_id}",
        final_weight=weight,
    )


def _weighter(seed: int = 42) -> DomainWeighter:
    return DomainWeighter(
        "config/source_weights.yaml",
        strategy="upsample",
        seed=seed,
    )


def test_upsample_keeps_documents_below_one():
    docs = [
        _doc("low-035", 0.35),
        _doc("low-064", 0.64),
        _doc("low-080", 0.80),
    ]
    out = list(_weighter().apply(iter(docs)))
    assert [d.doc_id for d in out] == ["low-035", "low-064", "low-080"]


def test_upsample_keeps_weight_one_once():
    docs = [_doc("exact", 1.0)]
    out = list(_weighter().apply(iter(docs)))
    assert len(out) == 1
    assert out[0].doc_id == "exact"


def test_upsample_never_emits_zero_copies_for_positive_weight():
    docs = [_doc("a", 0.05), _doc("b", 0.20), _doc("c", 0.99)]
    out = list(_weighter().apply(iter(docs)))
    assert {d.doc_id for d in out} == {"a", "b", "c"}


def test_upsample_weight_above_one_has_at_least_floor_copies():
    docs = [_doc("two", 2.0), _doc("three", 3.0)]
    out = list(_weighter().apply(iter(docs)))
    counts = {}
    for doc in out:
        counts[doc.doc_id] = counts.get(doc.doc_id, 0) + 1
    assert counts["two"] >= 2
    assert counts["three"] >= 3


def test_weighting_is_deterministic_for_a_fixed_seed():
    docs = [_doc(f"doc-{i}", 1.1 + (i % 7) / 10) for i in range(40)]
    first = [d.doc_id for d in _weighter(123).apply(iter(docs))]
    second = [d.doc_id for d in _weighter(123).apply(iter(docs))]
    assert first == second


def test_unknown_strategy_fails_loudly():
    weighter = DomainWeighter("config/source_weights.yaml", strategy="not-a-real-strategy")
    docs = [_doc("bad", 1.0)]
    try:
        list(weighter.apply(iter(docs)))
    except ValueError as exc:
        assert "Unknown weighting strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown strategy")
