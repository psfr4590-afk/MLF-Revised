"""
weighter.py — Applies source weights to produce a balanced, weighted corpus.

Strategies:
  upsample   — repeat high-weight docs (fractional weights → probabilistic)
  downsample — skip low-weight docs with probability (1 - weight)
  both       — downsample docs with weight < 1.0, upsample docs with weight > 1.0
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Iterator

import yaml

from pipeline.types import Document

log = logging.getLogger("weighter")


class DomainWeighter:
    def __init__(self, config_path: str | Path, strategy: str = "upsample", seed: int = 42):
        with open(config_path, encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f) or {}
        self._strategy = strategy
        self._rng = random.Random(seed)
        self.stats = {"docs_in": 0, "docs_out": 0, "total_weight": 0.0}

    def _repeat_count(self, weight: float) -> tuple[int, float]:
        whole = math.floor(weight)
        return whole, weight - whole

    def apply(self, docs: Iterator[Document]) -> Iterator[Document]:
        for doc in docs:
            self.stats["docs_in"] += 1
            w = max(float(doc.final_weight), 0.0)
            self.stats["total_weight"] += w
            if self._strategy == "upsample":
                whole = max(1, math.floor(w))
                frac = max(0.0, w - math.floor(w))
                for _ in range(whole):
                    self.stats["docs_out"] += 1
                    yield doc
                if w > 1.0 and frac > 0.0 and self._rng.random() < frac:
                    self.stats["docs_out"] += 1
                    yield doc
            elif self._strategy == "downsample":
                if w >= 1.0 or self._rng.random() < w:
                    self.stats["docs_out"] += 1
                    yield doc
            elif self._strategy == "both":
                if w < 1.0:
                    if self._rng.random() < w:
                        self.stats["docs_out"] += 1
                        yield doc
                else:
                    whole = math.floor(w)
                    frac = w - whole
                    for _ in range(max(1, whole)):
                        self.stats["docs_out"] += 1
                        yield doc
                    if frac > 0.0 and self._rng.random() < frac:
                        self.stats["docs_out"] += 1
                        yield doc
            else:
                raise ValueError(f"Unknown weighting strategy: {self._strategy!r}. Expected 'upsample', 'downsample', or 'both'.")

    def print_stats(self):
        s = self.stats
        log.info("Weighter | docs_in=%s docs_out=%s avg_weight=%.3f expansion=%.2fx", s["docs_in"], s["docs_out"], s["total_weight"] / max(s["docs_in"], 1), s["docs_out"] / max(s["docs_in"], 1))


_CONTENT_TYPE_KEYWORDS: list[tuple[list[str], str]] = [
    (["abstract", "introduction", "methodology", "conclusion", "references", "theorem", "lemma"], "research_paper"),
    (["def ", "class ", "fn ", "impl ", "function ", "#include", "import ", "from . import"], "source_code"),
    (["parameters", "returns", "example usage", "api reference", "install", "pip install"], "technical_doc"),
    (["dataset", "benchmark", "experiment", "evaluation", "accuracy", "f1 score"], "research_paper"),
    (["chapter ", "section ", "textbook", "exercise ", "problem set"], "textbook"),
    (["press release", "breaking news", "reported", "journalist"], "news_article"),
]


def reclassify_content_type(doc: Document) -> str:
    if doc.content_type not in ("unknown", ""):
        return doc.content_type
    text_lower = doc.text.lower()
    for keywords, ctype in _CONTENT_TYPE_KEYWORDS:
        if sum(1 for kw in keywords if kw in text_lower) >= 2:
            return ctype
    return "unknown"
