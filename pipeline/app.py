"""Backward-compatible import surface for the canonical pipeline orchestrator.

Historically Model Lab shipped two orchestration implementations. Keeping a
small compatibility module avoids breaking downstream imports while ensuring
there is exactly one implementation of pipeline behavior.
"""
from .orchestrator import Pipeline

__all__ = ["Pipeline"]
