"""
Shared data types for the pretrain pipeline.
Every stage consumes and/or produces Documents.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """Central data unit flowing through the pipeline."""

    doc_id: str
    url: str = ""
    source: str = ""
    text: str = ""
    title: str = ""
    language: str = "en"
    content_type: str = "unknown"
    code_language: str = ""
    domain: str = ""
    domain_weight: float = 1.0
    content_type_weight: float = 1.0
    quality_score: float = 1.0
    final_weight: float = 1.0
    stars: int = 0
    citations: int = 0
    word_count: int = 0
    char_count: int = 0
    crawl_depth: int = 0
    flagged: bool = False
    clean_action: str = ""
    clean_score: float = 0.0
    embedding: Optional[list[float]] = field(default=None, repr=False)
    dedup_cluster_id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.char_count:
            self.char_count = len(self.text)

    def to_jsonl(self) -> dict:
        d = self.__dict__.copy()
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
