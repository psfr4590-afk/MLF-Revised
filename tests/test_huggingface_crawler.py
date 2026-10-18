import pytest

import pipeline.crawler.huggingface_crawler as huggingface_module
from pipeline.crawler.huggingface_crawler import HuggingFaceCrawler


def test_row_text_fields_preserves_labeled_reasoning_fields() -> None:
    row = {
        "question": "What is 2 + 2?",
        "context": {"contexts": ["Basic arithmetic"]},
        "answer": "4",
    }

    text = HuggingFaceCrawler._row_text_fields(row, ["question", "context", "answer"])

    assert "question: What is 2 + 2?" in text
    assert 'context: {"contexts": ["Basic arithmetic"]}' in text
    assert "answer: 4" in text


def test_row_text_fields_skips_missing_fields() -> None:
    row = {"question": "What is 2 + 2?"}

    text = HuggingFaceCrawler._row_text_fields(row, ["question", "context", "answer"])

    assert text == "question: What is 2 + 2?"


def test_crawl_records_retrieval_provenance_without_weight_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        huggingface_module,
        "load_dataset",
        lambda *args, **kwargs: [{"question": "What is 2 + 2?", "answer": "4"}],
    )
    crawler = HuggingFaceCrawler(
        {
            "huggingface": {
                "datasets": [
                    {
                        "repo": "example/dataset",
                        "revision": "0123456",
                        "split": "train",
                        "text_fields": ["question", "answer"],
                        "max_docs": 1,
                    }
                ]
            }
        },
        weight_lookup=None,
        signal_tracker=None,
    )

    doc = next(crawler.crawl())

    assert doc.meta["source_identity"]["revision"] == "0123456"
    assert doc.meta["retrieval_provenance"]["identity_type"] == "canonical_record"
    assert len(doc.meta["retrieval_provenance"]["record_sha256"]) == 64


def test_required_hf_dataset_preserves_row_license(monkeypatch) -> None:
    rows = iter(
        [
            {"content": "licensed software engineering example", "license": "MIT"},
            {"content": "missing license must be rejected"},
        ]
    )

    monkeypatch.setattr(huggingface_module, "load_dataset", lambda *args, **kwargs: rows)
    crawler = HuggingFaceCrawler(
        {
            "huggingface": {
                "datasets": [
                    {
                        "repo": "example/curated",
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                        "split": "train",
                        "text_field": "content",
                        "max_docs": 10,
                        "required": True,
                        "require_license": True,
                    }
                ]
            }
        },
        None,
        None,
    )

    docs = list(crawler.crawl())

    assert len(docs) == 1
    assert docs[0].meta["license"] == "mit"
    assert docs[0].meta["license_status"] == "mit"
    assert docs[0].meta["rights_status"] == "review_required"


def test_required_hf_dataset_failure_is_fatal(monkeypatch) -> None:
    def fake_load_dataset(_dataset_id, **_kwargs):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr(huggingface_module, "load_dataset", fake_load_dataset)
    crawler = HuggingFaceCrawler(
        {
            "huggingface": {
                "datasets": [
                    {
                        "repo": "example/curated",
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                        "split": "train",
                        "text_field": "content",
                        "max_docs": 10,
                        "required": True,
                    }
                ]
            }
        },
        None,
        None,
    )

    with pytest.raises(RuntimeError, match="Required Hugging Face dataset crawl failed"):
        list(crawler.crawl())
