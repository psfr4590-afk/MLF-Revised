from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.types import Document
from pipeline.weighter.weighter import DomainWeighter, reclassify_content_type


class TestWeightLookup:
    def _config(self, tmp_path: Path) -> Path:
        path = tmp_path / "weights.yaml"
        path.write_text("weights: {}\n", encoding="utf-8")
        return path

    def _doc(self, weight: float) -> Document:
        return Document(doc_id=f"doc-{weight}", text="content", final_weight=weight)


def test_repeat_count_splits_whole_and_fraction(tmp_path):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path))
    assert weighter._repeat_count(3.75) == (3, 0.75)
    assert weighter._repeat_count(0.25) == (0, 0.25)


def test_upsample_keeps_zero_weight_once_and_expands_whole_weight(tmp_path):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path), strategy="upsample", seed=42)
    docs = list(weighter.apply(iter([TestWeightLookup()._doc(0.0), TestWeightLookup()._doc(2.0)])))
    assert [d.final_weight for d in docs] == [0.0, 2.0, 2.0]
    assert weighter.stats == {"docs_in": 2, "docs_out": 3, "total_weight": 2.0}


def test_upsample_fraction_is_probabilistic_but_seeded(tmp_path):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path), strategy="upsample", seed=1)
    docs = list(weighter.apply(iter([TestWeightLookup()._doc(2.5)])))
    assert len(docs) == 3
    assert weighter.stats["docs_out"] == 3


def test_downsample_handles_zero_and_full_weights(tmp_path):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path), strategy="downsample", seed=42)
    docs = list(weighter.apply(iter([TestWeightLookup()._doc(0.0), TestWeightLookup()._doc(1.0), TestWeightLookup()._doc(2.0)])))
    assert [d.final_weight for d in docs] == [1.0, 2.0]
    assert weighter.stats["docs_in"] == 3
    assert weighter.stats["docs_out"] == 2


def test_downsample_fraction_uses_seeded_rng(tmp_path):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path), strategy="downsample", seed=1)
    docs = list(weighter.apply(iter([TestWeightLookup()._doc(0.5)])))
    assert len(docs) == 1


def test_both_downsamples_low_weights_and_upsamples_high_weights(tmp_path):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path), strategy="both", seed=1)
    docs = list(
        weighter.apply(
            iter(
                [
                    TestWeightLookup()._doc(0.5),
                    TestWeightLookup()._doc(1.0),
                    TestWeightLookup()._doc(2.5),
                ]
            )
        )
    )
    # seed=1 yields 0.134... for the low-weight decision and 0.847... for
    # the fractional high-weight decision, so the 2.5 document is repeated
    # only for its two whole-weight copies.
    assert [d.final_weight for d in docs] == [0.5, 1.0, 2.5, 2.5]


def test_negative_weight_is_clamped_to_zero(tmp_path):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path), strategy="downsample", seed=42)
    docs = list(weighter.apply(iter([TestWeightLookup()._doc(-5.0)])))
    assert docs == []
    assert weighter.stats["total_weight"] == 0.0


def test_unknown_strategy_fails_closed(tmp_path):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path), strategy="mystery")
    with pytest.raises(ValueError, match="Unknown weighting strategy"):
        list(weighter.apply(iter([TestWeightLookup()._doc(1.0)])))


def test_print_stats_does_not_divide_by_zero(tmp_path, caplog):
    weighter = DomainWeighter(TestWeightLookup()._config(tmp_path))
    with caplog.at_level("INFO", logger="weighter"):
        weighter.print_stats()
    assert "avg_weight=0.000" in caplog.text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("abstract introduction methodology", "research_paper"),
        ("def example():\nclass Demo:", "source_code"),
        ("parameters returns API reference", "technical_doc"),
        ("dataset benchmark accuracy", "research_paper"),
        ("chapter one textbook exercise", "textbook"),
        ("press release breaking news", "news_article"),
        ("one unrelated sentence", "unknown"),
    ],
)
def test_reclassify_content_type_detects_known_keyword_groups(text, expected):
    doc = Document(doc_id="x", text=text, content_type="unknown")
    assert reclassify_content_type(doc) == expected


def test_reclassify_preserves_explicit_content_type():
    doc = Document(doc_id="x", text="abstract introduction methodology", content_type="technical_doc")
    assert reclassify_content_type(doc) == "technical_doc"
