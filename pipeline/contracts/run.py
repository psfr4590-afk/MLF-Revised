"""Canonical pipeline document contract.

Model Lab keeps ``pipeline.types.Document`` as the single source of truth.
This compatibility module preserves the public contract namespace used by the
1.3.0 architecture without maintaining a second, drifting dataclass.
"""

from pipeline.types import Document

__all__ = ["Document"]
