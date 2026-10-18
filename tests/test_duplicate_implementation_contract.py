"""Regression tests preventing duplicate production implementations."""

from pipeline.cleaner.cleaner import Cleaner as CanonicalCleaner
from pipeline.cleaner.cleaner import load_config as canonical_load_config
from pipeline.filtering.quality_filter import Cleaner as CompatibilityCleaner
from pipeline.filtering.quality_filter import load_config as compatibility_load_config
from pipeline.filtering.reporter import DomainWeighter as CompatibilityWeighter
from pipeline.filtering.reporter import reclassify_content_type as compatibility_reclassify
from pipeline.weighter.weighter import DomainWeighter as CanonicalWeighter
from pipeline.weighter.weighter import reclassify_content_type as canonical_reclassify


def test_quality_filter_is_compatibility_shim():
    assert CompatibilityCleaner is CanonicalCleaner
    assert compatibility_load_config is canonical_load_config


def test_reporter_is_weighting_compatibility_shim():
    assert CompatibilityWeighter is CanonicalWeighter
    assert compatibility_reclassify is canonical_reclassify
