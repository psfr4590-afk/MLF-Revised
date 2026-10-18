"""
source_scorer.py — Signal quality scoring for crawled pages and domains.

Scores a raw (url, text) pair 0.0–1.0. The crawler uses this to:
  1. Assign content_type and weights to each document.
  2. Maintain a rolling signal score per domain and abandon low-signal domains
     after min_pages_before_score pages.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from urllib.parse import urlparse

import yaml

_URL_TYPE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"arxiv\.org/(abs|pdf|html)/\d"), "research_paper"),
    (re.compile(r"aclanthology\.org/\d{4}\.\w+-\w+"), "research_paper"),
    (re.compile(r"openreview\.net/forum"), "research_paper"),
    (re.compile(r"semanticscholar\.org/paper"), "research_paper"),
    (re.compile(r"ncbi\.nlm\.nih\.gov/(pmc|pubmed)"), "research_paper"),
    (re.compile(r"github\.com/[^/]+/[^/]+/(blob|tree)/"), "source_code"),
    (re.compile(r"github\.com/[^/]+/[^/]+/?$"), "source_code"),
    (re.compile(r"\.(py|rs|go|c|cpp|h|ts|js|java|rb|swift|zig|lua|ml|hs)$"), "source_code"),
    (re.compile(r"readthedocs\.(io|org)"), "technical_doc"),
    (re.compile(r"docs\.(python|rust-lang|golang|pytorch|tensorflow|numpy)\."), "technical_doc"),
    (re.compile(r"developer\.mozilla\.org"), "technical_doc"),
    (re.compile(r"wikipedia\.org/wiki/"), "encyclopedia"),
    (re.compile(r"stackoverflow\.com/questions/\d"), "technical_qa"),
    (re.compile(r"stackexchange\.com/questions/\d"), "technical_qa"),
    (re.compile(r"distill\.pub"), "research_paper"),
    (re.compile(r"paulgraham\.com"), "long_form_essay"),
    (re.compile(r"(huggingface|lilianweng)"), "research_blog"),
    (re.compile(r"(medium|substack)\.com"), "blog_post"),
    (re.compile(r"(reddit|news\.ycombinator)\.com"), "forum_post"),
    (re.compile(r"(twitter|instagram|tiktok|facebook)\.com"), "social_media"),
]

_TEXT_TYPE_RULES: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r"\b(abstract|introduction|related work|methodology|conclusion|references)\b", re.I), "research_paper", 0.003),
    (re.compile(r"\b(def |class |fn |impl |func |function |#include|import |from .+ import)\b"), "source_code", 0.005),
    (re.compile(r"\b(theorem|lemma|proof|corollary|proposition)\b", re.I), "research_paper", 0.002),
    (re.compile(r"\b(dataset|benchmark|baseline|ablation|evaluation|epoch|batch_size)\b", re.I), "research_paper", 0.004),
    (re.compile(r"\bparameters?\b.*\bmodel\b|\bmodel\b.*\bparameters?\b", re.I), "research_paper", 0.002),
]

_REWARD_PATTERNS = [
    re.compile(r"\bfigure \d+\b", re.I),
    re.compile(r"\b(theorem|lemma|proof|corollary|proposition)\b", re.I),
    re.compile(r"\b(algorithm|complexity|O\(n\))\b", re.I),
    re.compile(r"\b(dataset|experiment|baseline|benchmark|ablation)\b", re.I),
    re.compile(r"\b(def |class |fn |impl |func )\b"),
    re.compile(r"\breferences?\b", re.I),
    re.compile(r"\b(section|subsection|appendix)\s+\d", re.I),
    re.compile(r"\b(loss|gradient|backprop|optimizer|epoch|batch)\b", re.I),
    re.compile(r"\b(precision|recall|F1|accuracy|BLEU|ROUGE|perplexity)\b", re.I),
    re.compile(r"\b(assert|unittest|pytest|test_)\b"),
    re.compile(r"\b(API|REST|endpoint|schema|interface)\b", re.I),
    re.compile(r"[A-Z][a-z]+\s+et\s+al\.", re.I),
    re.compile(r"\d{4}\.\d{4,5}"),
]

_PENALTY_PATTERNS = [
    re.compile(r"click here", re.I),
    re.compile(r"sign up (now|today|free)", re.I),
    re.compile(r"limited time offer", re.I),
    re.compile(r"subscribe (now|to our newsletter)", re.I),
    re.compile(r"\bcookie policy\b", re.I),
    re.compile(r"\baffiliate\b", re.I),
    re.compile(r"\bsponsored\b", re.I),
    re.compile(r"all rights reserved", re.I),
    re.compile(r"\bprivacy policy\b", re.I),
    re.compile(r"(buy now|add to cart|checkout)", re.I),
    re.compile(r"[^\w\s]{8,}"),
    re.compile(r"(\b\w+\b)( \1){4,}", re.I),
]

_CODE_LANG_MAP = {
    ".py": "python", ".rs": "rust", ".go": "go", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".ts": "typescript",
    ".js": "javascript", ".java": "java", ".rb": "ruby", ".swift": "swift",
    ".zig": "zig", ".lua": "lua", ".ml": "ocaml", ".hs": "haskell",
    ".jl": "julia", ".r": "r", ".sql": "sql", ".sh": "bash", ".bash": "bash",
}


def classify_url(url: str) -> str:
    for pat, ctype in _URL_TYPE_RULES:
        if pat.search(url):
            return ctype
    return "unknown"


def classify_text(text: str) -> str:
    tokens = max(len(text.split()), 1)
    for pat, ctype, threshold in _TEXT_TYPE_RULES:
        if len(pat.findall(text)) / tokens >= threshold:
            return ctype
    return "unknown"


def detect_code_language(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext, lang in _CODE_LANG_MAP.items():
        if path.endswith(ext):
            return lang
    return ""


def score_document(url: str, text: str) -> float:
    if not text or len(text.split()) < 30:
        return 0.1
    tokens = len(text.split())
    chars = len(text)
    url_type = classify_url(url)
    base_scores = {
        "research_paper": 0.9, "source_code": 0.85, "technical_doc": 0.8,
        "encyclopedia": 0.7, "technical_qa": 0.7, "research_blog": 0.7,
        "long_form_essay": 0.65, "technical_blog": 0.6, "blog_post": 0.45,
        "forum_post": 0.3, "social_media": 0.05, "unknown": 0.4,
    }
    score = base_scores.get(url_type, 0.4)
    score += sum(1 for p in _REWARD_PATTERNS if p.search(text)) * 0.03
    score -= sum(1 for p in _PENALTY_PATTERNS if p.search(text)) * 0.08
    if tokens > 500:
        score += 0.05
    if tokens > 2000:
        score += 0.05
    alpha_ratio = sum(c.isalpha() for c in text) / max(chars, 1)
    if alpha_ratio < 0.5:
        score -= 0.1
    unique_ratio = len(set(text.lower().split())) / tokens
    if unique_ratio < 0.2:
        score -= 0.15
    digit_ratio = sum(c.isdigit() for c in text) / max(chars, 1)
    if digit_ratio > 0.3:
        score -= 0.1
    return max(0.0, min(1.0, score))


class DomainSignalTracker:
    def __init__(self, cfg: dict):
        self._gate = cfg.get("signal_gate", {})
        self._enabled = self._gate.get("enabled", True)
        self._min_pages = self._gate.get("min_pages_before_score", 10)
        self._threshold = self._gate.get("signal_threshold", 0.35)
        self._window = self._gate.get("window", 20)
        self._scores: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=self._window))
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._abandoned: set[str] = set()

    def record(self, domain: str, score: float) -> None:
        self._scores[domain].append(score)
        self._counts[domain] += 1

    def should_abandon(self, domain: str) -> bool:
        if not self._enabled or domain in self._abandoned:
            return domain in self._abandoned
        if self._counts[domain] < self._min_pages:
            return False
        scores = self._scores[domain]
        if scores and sum(scores) / len(scores) < self._threshold:
            self._abandoned.add(domain)
            return True
        return False

    def avg_score(self, domain: str) -> float:
        scores = self._scores[domain]
        return sum(scores) / len(scores) if scores else 0.0

    def abandoned_domains(self) -> set[str]:
        return set(self._abandoned)

    def snapshot(self) -> list[dict]:
        rows = []
        for domain, scores in self._scores.items():
            if scores:
                rows.append({"domain": domain, "avg_score": self.avg_score(domain), "count": self._counts[domain], "abandoned": domain in self._abandoned})
        return sorted(rows, key=lambda r: (-r["avg_score"], r["domain"]))


class SourceWeightLookup:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self._domain_rules = cfg.get("domain_weights", [])
        self._type_weights = cfg.get("content_type_weights", {})
        self._lang_weights = cfg.get("code_language_weights", {})
        self._quality_cfg = cfg.get("quality_heuristics", {})
        self._reward_pats = [re.compile(p, re.I) for p in self._quality_cfg.get("reward_patterns", [])]
        self._penalty_pats = [re.compile(p, re.I) for p in self._quality_cfg.get("penalty_patterns", [])]
        self._signal_gate = cfg.get("signal_gate", {})

    def domain_weight(self, domain: str) -> tuple[float, str]:
        import fnmatch
        for rule in self._domain_rules:
            pat = rule.get("pattern", "*")
            if pat == "*" or fnmatch.fnmatch(domain, pat) or pat in domain:
                return float(rule.get("weight", 1.0)), rule.get("category", "general_web")
        return 0.8, "general_web"

    def content_type_weight(self, ctype: str) -> float:
        return float(self._type_weights.get(ctype, self._type_weights.get("unknown", 0.8)))

    def code_language_weight(self, lang: str) -> float:
        return float(self._lang_weights.get(lang, self._lang_weights.get("unknown", 1.0)))

    def quality_score(self, text: str) -> float:
        words = text.split()
        n = max(len(words), 1)
        chars = max(len(text), 1)
        cfg = self._quality_cfg
        score = 0.5
        if n < cfg.get("min_word_count", 50):
            return 0.1
        if n > cfg.get("max_word_count", 100000):
            score -= 0.1
        avg_wl = sum(len(w) for w in words) / n
        if not (cfg.get("min_avg_word_length", 3.5) <= avg_wl <= cfg.get("max_avg_word_length", 12.0)):
            score -= 0.15
        unique_ratio = len(set(w.lower() for w in words)) / n
        if unique_ratio < cfg.get("min_unique_word_ratio", 0.2):
            score -= 0.2
        digit_ratio = sum(c.isdigit() for c in text) / chars
        if digit_ratio > cfg.get("max_digit_ratio", 0.3):
            score -= 0.1
        alpha_ratio = sum(c.isalpha() for c in text) / chars
        if alpha_ratio < cfg.get("min_alpha_ratio", 0.6):
            score -= 0.15
        score += sum(0.05 for p in self._reward_pats if p.search(text))
        score -= sum(0.08 for p in self._penalty_pats if p.search(text))
        return max(0.05, min(1.0, score))

    def signal_gate_config(self) -> dict:
        return self._signal_gate
