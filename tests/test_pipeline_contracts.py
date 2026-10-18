from pipeline.contracts.run import Document as ContractDocument
from pipeline.types import Document


def test_contract_document_is_canonical_alias():
    assert ContractDocument is Document


def test_document_jsonl_round_trip_drops_embedding():
    original = Document(
        doc_id="doc-1",
        url="https://example.com",
        text="hello world",
        embedding=[0.1, 0.2],
        meta={"source": "test"},
    )

    payload = original.to_jsonl()
    restored = ContractDocument.from_dict(payload)

    assert "embedding" not in payload
    assert restored.doc_id == original.doc_id
    assert restored.text == original.text
    assert restored.meta == original.meta
    assert restored.embedding is None
