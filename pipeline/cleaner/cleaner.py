"""cleaner.py — Pretraining data cleaner stage.

Pipeline: HTML strip → unicode normalize → length gate → exact-hash dedup
          → refusal/restriction filter → pass/drop/flag/redact
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

log = logging.getLogger("cleaner")


class Action(str, Enum):
    DROP = "drop"
    FLAG = "flag"
    REDACT = "redact"


@dataclass
class PatternGroup:
    name: str
    enabled: bool
    weight: float
    patterns: list[re.Pattern]


@dataclass
class CleanerConfig:
    action: Action
    score_threshold: float
    min_doc_length: int
    max_doc_length: int
    log_enabled: bool
    log_level: str
    log_file: str
    log_matched_text: bool
    log_doc_preview: int
    html_enabled: bool
    html_parser: str
    unicode_enabled: bool
    unicode_form: str
    strip_control_chars: bool
    dedup_enabled: bool
    dedup_method: str
    refusal_enabled: bool
    pattern_groups: list[PatternGroup]
    instant_drop_enabled: bool
    instant_drop_patterns: list[re.Pattern]


def _compile(raw: list[str]) -> list[re.Pattern]:
    out = []
    for p in raw:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error as exc:
            log.warning("Bad regex %r: %s", p, exc)
    return out


def load_config(path: str | Path) -> CleanerConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    g, lg, h, u, d, rf = (raw.get(k, {}) for k in ("general", "logging", "html", "unicode", "dedup", "refusal_filter"))
    groups = [PatternGroup(name=n, enabled=c.get("enabled", True), weight=float(c.get("weight", 1.0)), patterns=_compile(c.get("patterns", []))) for n, c in rf.get("pattern_groups", {}).items()]
    idp = rf.get("instant_drop_patterns", {})
    return CleanerConfig(
        action=Action(g.get("action", "drop")), score_threshold=float(g.get("score_threshold", 0.12)),
        min_doc_length=int(g.get("min_doc_length", 80)), max_doc_length=int(g.get("max_doc_length", 2_000_000)),
        log_enabled=lg.get("enabled", True), log_level=lg.get("level", "INFO"), log_file=lg.get("log_file", "output/logs/cleaner.log"),
        log_matched_text=lg.get("log_matched_text", True), log_doc_preview=int(lg.get("log_doc_preview", 120)),
        html_enabled=h.get("enabled", True), html_parser=h.get("parser", "lxml"), unicode_enabled=u.get("enabled", True),
        unicode_form=u.get("normalize_form", "NFKC"), strip_control_chars=u.get("strip_control_chars", True),
        dedup_enabled=d.get("enabled", True), dedup_method=d.get("method", "exact_hash"), refusal_enabled=rf.get("enabled", True),
        pattern_groups=groups, instant_drop_enabled=idp.get("enabled", True), instant_drop_patterns=_compile(idp.get("patterns", [])),
    )


@dataclass
class CleanResult:
    kept: bool
    action: str
    text: str
    score: float = 0.0
    hits: list[dict] = field(default_factory=list)
    doc_id: str = ""


class Cleaner:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self._seen: set[str] = set()
        self._mtime = self.config_path.stat().st_mtime
        self.stats = {"total": 0, "kept": 0, "dropped_refusal": 0, "dropped_short": 0, "dropped_long": 0, "dropped_dedup": 0, "flagged": 0, "redacted": 0}

    def reload(self):
        self.cfg = load_config(self.config_path)
        self._mtime = self.config_path.stat().st_mtime

    def _maybe_reload(self):
        try:
            if self.config_path.stat().st_mtime != self._mtime:
                self.reload()
        except OSError:
            pass

    def _strip_html(self, text: str) -> str:
        if not self.cfg.html_enabled:
            return text
        if BS4_AVAILABLE:
            soup = BeautifulSoup(text, self.cfg.html_parser)
            for tag in soup(["script", "style", "nav", "footer", "aside", "header", "form", "noscript", "iframe"]):
                tag.decompose()
            return soup.get_text(separator=" ")
        return re.sub(r"<[^>]+>", " ", text)

    def _normalize(self, text: str) -> str:
        if not self.cfg.unicode_enabled:
            return text
        text = unicodedata.normalize(self.cfg.unicode_form, text)
        if self.cfg.strip_control_chars:
            text = "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf") or c in ("\n", "\t"))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _is_duplicate(self, text: str) -> bool:
        if not self.cfg.dedup_enabled:
            return False
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def _score(self, text: str) -> tuple[float, list[dict]]:
        hits, weighted = [], 0.0
        tokens = max(len(text.split()), 1)
        if self.cfg.instant_drop_enabled:
            for pat in self.cfg.instant_drop_patterns:
                m = pat.search(text)
                if m:
                    return 999.0, [{"group": "instant_drop", "pattern": pat.pattern, "span": m.span(), "matched": m.group(0) if self.cfg.log_matched_text else ""}]
        for grp in self.cfg.pattern_groups:
            if not grp.enabled:
                continue
            for pat in grp.patterns:
                for m in pat.finditer(text):
                    hits.append({"group": grp.name, "pattern": pat.pattern, "span": m.span(), "matched": m.group(0) if self.cfg.log_matched_text else ""})
                    weighted += grp.weight
        return weighted / tokens, hits

    def _redact(self, text: str, hits: list[dict]) -> str:
        chars = list(text)
        for h in sorted(hits, key=lambda x: x["span"][0], reverse=True):
            s, e = h["span"]
            chars[s:e] = list("[REMOVED]")
        return "".join(chars)

    def clean(self, text: str, doc_id: str = "") -> CleanResult:
        self._maybe_reload(); self.stats["total"] += 1; cfg = self.cfg
        text = self._normalize(self._strip_html(text))
        if len(text) < cfg.min_doc_length:
            self.stats["dropped_short"] += 1
            return CleanResult(False, "too_short", "", doc_id=doc_id)
        if len(text) > cfg.max_doc_length:
            self.stats["dropped_long"] += 1
            return CleanResult(False, "too_long", "", doc_id=doc_id)
        if self._is_duplicate(text):
            self.stats["dropped_dedup"] += 1
            return CleanResult(False, "duplicate", "", doc_id=doc_id)
        score, hits = self._score(text) if cfg.refusal_enabled else (0.0, [])
        if score >= cfg.score_threshold:
            if cfg.action == Action.DROP:
                self.stats["dropped_refusal"] += 1
                return CleanResult(False, "dropped", "", score, hits, doc_id)
            if cfg.action == Action.FLAG:
                self.stats["flagged"] += 1
                return CleanResult(True, "flagged", text, score, hits, doc_id)
            self.stats["redacted"] += 1
            return CleanResult(True, "redacted", self._redact(text, hits), score, hits, doc_id)
        self.stats["kept"] += 1
        return CleanResult(True, "kept", text, score, hits, doc_id)

    def print_stats(self):
        s = self.stats; total = max(s["total"], 1)
        log.info("Cleaner stats | total=%s kept=%s (%.1f%%) drop_refusal=%s drop_short=%s drop_long=%s drop_dedup=%s flagged=%s redacted=%s", s["total"], s["kept"], s["kept"] / total * 100, s["dropped_refusal"], s["dropped_short"], s["dropped_long"], s["dropped_dedup"], s["flagged"], s["redacted"])

    def reset_dedup(self):
        self._seen.clear()
