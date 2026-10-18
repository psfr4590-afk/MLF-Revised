from scripts.evaluate_model import _generation_metrics, _perplexity, _qualitative, _summary


def test_perplexity_is_exp_loss():
    assert round(_perplexity(3.24), 3) == 25.534


def test_generation_metrics_capture_repetition_and_diversity():
    metrics = _generation_metrics([1, 1, 1, 2, 3, 4], "a a a b c d")
    assert metrics["repetition_rate"] > 0
    assert metrics["distinct_1"] < 1
    assert metrics["distinct_2"] > 0


def test_qualitative_does_not_claim_coherence_from_nonempty_text_alone():
    metrics = _generation_metrics([1] * 20, "hello hello hello hello")
    quality = _qualitative("hello hello hello hello", metrics)
    assert quality["language"] == "PASS"
    assert quality["coherence"] == "LIMITED"


def test_summary_covers_all_domains():
    domains = {
        "general": [{"quality": {"language": "PASS", "coherence": "OBSERVED"}, "metrics": {"repetition_rate": 0.1, "distinct_1": 0.9, "distinct_2": 0.8}}],
        "technical": [{"quality": {"language": "PASS", "coherence": "OBSERVED"}, "metrics": {"repetition_rate": 0.2, "distinct_1": 0.8, "distinct_2": 0.7}}],
    }
    summary = _summary(domains)
    assert summary["language"] == "PASS"
    assert summary["domain_behavior"] == "PASS"
    assert summary["domain_count"] == 2
