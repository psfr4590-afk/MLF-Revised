"""Backward-compatible import shim for weighting utilities.

The canonical implementation lives in :mod:`pipeline.weighter.weighter`.
This module remains for older imports while avoiding a second implementation.
"""

from pipeline.weighter.weighter import DomainWeighter, reclassify_content_type

__all__ = ["DomainWeighter", "reclassify_content_type"]
