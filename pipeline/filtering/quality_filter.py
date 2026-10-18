"""Backward-compatible import shim for the canonical cleaner stage.

The production implementation lives in :mod:`pipeline.cleaner.cleaner`.
Keep this module as a compatibility surface for older callers rather than
maintaining a second implementation that can drift from the pipeline.
"""

from pipeline.cleaner.cleaner import (
    Action,
    CleanResult,
    Cleaner,
    CleanerConfig,
    PatternGroup,
    load_config,
)

__all__ = [
    "Action",
    "CleanResult",
    "Cleaner",
    "CleanerConfig",
    "PatternGroup",
    "load_config",
]
