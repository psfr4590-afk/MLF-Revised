"""
semantic_dedup.py — Near-duplicate removal using sentence-transformer embeddings + FAISS.

Strategy:
  1. Embed each document (first N chars for speed).
  2. Build a FAISS flat index (exact cosine) or IVF (approximate, for >1M docs).
  3. For each doc, query k-nearest neighbors.
  4. If any neighbor has cosine similarity >= threshold, mark as duplicate,
     keeping whichever has the higher final_weight.

Memory: all-MiniLM-L6-v2 is 384-dim float32 = ~1.5KB/doc.
        1M docs ≈ 1.5GB; fine for your 8GB RAM.
        If you hit limits, switch index_type to ivf in config.

Offline behavior:
  - sentence-transformers and faiss-cpu are required runtime dependencies.
  - mode=auto (default) never downloads a model. It uses embeddings only when
    the configured model is already available locally; otherwise it falls back
    to deterministic token-gram similarity.
  - mode=fallback explicitly selects the deterministic token-gram implementation
    for tests or deliberately degraded operation.
  - mode=embedding requires the required packages and a locally available model
    unless allow_model_download=true is explicitly configured.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

import numpy as np

from pipeline.types import Document

log = logging.getLogger("embedder")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    ST_AVAILABLE = False
    log.warning("sentence-transformers not installed; semantic dedup will use fallback")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    log.warning("faiss-cpu not installed; falling back to brute-force cosine")


def _cosine_brute(matrix: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force cosine similarity when FAISS is unavailable."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms
    q_normed = query / (np.linalg.norm(query) + 1e-10)
    sims = normed @ q_normed
    top_k = np.argsort(sims)[::-1][:k]
    return sims[top_k], top_k


class SemanticDeduplicator:

    def __init__(self, cfg: dict):
        self._cfg       = cfg
        self._model_name = cfg.get("model", "all-MiniLM-L6-v2")
        self._batch     = cfg.get("batch_size", 256)
        self._threshold = cfg.get("similarity_threshold", 0.92)
        self._chunk     = cfg.get("chunk_size", 512)
        self._idx_type  = cfg.get("index_type", "flat")
        self._nlist     = cfg.get("ivf_nlist", 100)
        self._mode      = str(cfg.get("mode", "auto")).lower()
        self._allow_model_download = bool(cfg.get("allow_model_download", False))
        self._model_path = cfg.get("model_path")
        self._model     = None
        self._index     = None
        self._dim       = 0
        self._docs:     list[Document] = []
        self._embeddings: list[np.ndarray] = []
        self.stats = {"total": 0, "kept": 0, "dropped": 0}

    @staticmethod
    def _fallback_similarity(a: str, b: str) -> float:
        """Dependency-light token-gram similarity used when embeddings are unavailable."""
        def grams(text: str) -> set[str]:
            tokens = re.findall(r"\w+", text.lower())
            return {" ".join(tokens[i:i+3]) for i in range(max(0, len(tokens) - 2))} or set(tokens)
        ga, gb = grams(a), grams(b)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)

    def _fallback_run(self, docs: list[Document]) -> list[Document]:
        kept: list[Document] = []
        for doc in docs:
            duplicate = False
            for index, prior in enumerate(kept):
                if self._fallback_similarity(doc.text, prior.text) >= self._threshold:
                    duplicate = True
                    if doc.final_weight > prior.final_weight:
                        kept[index] = doc
                    break
            if not duplicate:
                kept.append(doc)
        self.stats["total"] += len(docs)
        self.stats["kept"] += len(kept)
        self.stats["dropped"] += len(docs) - len(kept)
        log.warning("Semantic dedup fallback active: embeddings unavailable")
        return kept

    def _fallback_stream(self, docs: Iterator[Document], buffer_size: int) -> Iterator[Document]:
        """Consume an iterator in bounded batches without materializing the input."""
        reps: list[Document] = []
        buffer: list[Document] = []
        total = 0

        def process(batch: list[Document]) -> None:
            nonlocal total
            for doc in batch:
                total += 1
                duplicate = False
                for index, prior in enumerate(reps):
                    if self._fallback_similarity(doc.text, prior.text) >= self._threshold:
                        duplicate = True
                        if doc.final_weight > prior.final_weight:
                            reps[index] = doc
                        break
                if not duplicate:
                    reps.append(doc)

        for doc in docs:
            buffer.append(doc)
            if len(buffer) >= buffer_size:
                process(buffer)
                buffer.clear()
        if buffer:
            process(buffer)

        self.stats["total"] += total
        self.stats["kept"] += len(reps)
        self.stats["dropped"] += total - len(reps)
        log.warning("Semantic dedup fallback active: embeddings unavailable")
        log.info("Semantic dedup stream: %d → %d (-%d)", total, len(reps), total - len(reps))
        yield from reps

    def _load_model(self):
        if self._mode == "fallback":
            raise RuntimeError("semantic dedup configured for fallback mode")
        if not ST_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")
        if self._model is None:
            model_ref = self._model_path or self._model_name
            log.info("Loading sentence-transformer: %s", model_ref)
            kwargs = {}
            if not self._allow_model_download:
                kwargs["local_files_only"] = True
            self._model = SentenceTransformer(model_ref, **kwargs)
            dimension_getter = getattr(self._model, "get_embedding_dimension", None)
            if dimension_getter is None:
                dimension_getter = self._model.get_sentence_embedding_dimension
            self._dim = dimension_getter()
            log.info("Embedding dim: %s", self._dim)

    def _ensure_embedding_or_fallback(self) -> bool:
        if self._mode == "fallback":
            return False
        if not ST_AVAILABLE or not FAISS_AVAILABLE:
            missing = []
            if not ST_AVAILABLE:
                missing.append("sentence-transformers")
            if not FAISS_AVAILABLE:
                missing.append("faiss-cpu")
            raise RuntimeError(
                "Semantic dedup dependencies are required but unavailable: "
                + ", ".join(missing)
            )
        try:
            self._load_model()
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            if self._mode == "embedding":
                raise RuntimeError(
                    "Semantic dedup embedding mode requires a locally available "
                    f"model ({self._model_path or self._model_name}). "
                    "Cache/provide the model, or set allow_model_download=true explicitly."
                ) from exc
            log.warning("Semantic embedding model unavailable; using deterministic offline fallback: %s", exc)
            return False

    def _embed(self, texts: list[str]) -> np.ndarray:
        chunks = [t[:self._chunk] for t in texts]
        vecs   = self._model.encode(
            chunks,
            batch_size=self._batch,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def _build_index(self, matrix: np.ndarray) -> "faiss.Index":
        d = matrix.shape[1]
        if self._idx_type == "ivf" and FAISS_AVAILABLE and faiss is not None:
            nlist = min(self._nlist, max(1, matrix.shape[0] // 10))
            quantizer = faiss.IndexFlatIP(d)
            index     = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(matrix)
            index.add(matrix)
            index.nprobe = min(10, nlist)
        elif FAISS_AVAILABLE and faiss is not None:
            index = faiss.IndexFlatIP(d)
            index.add(matrix)
        else:
            index = None
        return index

    def run(self, docs: list[Document]) -> list[Document]:
        """Deduplicate a list of documents and return the subset to keep."""
        if not docs:
            return docs

        if not self._ensure_embedding_or_fallback():
            return self._fallback_run(docs)

        self.stats["total"] += len(docs)
        log.info("Embedding %d docs ...", len(docs))
        matrix = self._embed([d.text for d in docs])

        log.info("Building FAISS index ...")
        index  = self._build_index(matrix)
        is_dup = [False] * len(docs)
        k = min(5, len(docs))

        for i in range(len(docs)):
            if is_dup[i]:
                continue
            q = matrix[i:i+1]
            if FAISS_AVAILABLE and index is not None:
                sims, idxs = index.search(q, k + 1)
                sims = sims[0]
                idxs = idxs[0]
            else:
                sims, idxs = _cosine_brute(matrix, q[0], k + 1)

            for sim, j in zip(sims, idxs, strict=True):
                j = int(j)
                if j <= i or j >= len(docs):
                    continue
                if float(sim) >= self._threshold:
                    if docs[i].final_weight >= docs[j].final_weight:
                        is_dup[j] = True
                        log.debug("DUP drop [%d] %s sim=%.3f vs [%d] %s", j, docs[j].url[:60], sim, i, docs[i].url[:60])
                    else:
                        is_dup[i] = True
                        log.debug("DUP drop [%d] %s sim=%.3f vs [%d] %s", i, docs[i].url[:60], sim, j, docs[j].url[:60])
                    break

        kept = [d for d, dup in zip(docs, is_dup, strict=True) if not dup]
        dropped = len(docs) - len(kept)
        self.stats["kept"] += len(kept)
        self.stats["dropped"] += dropped
        log.info("Semantic dedup: %d → %d (-%d)", len(docs), len(kept), dropped)
        return kept

    def stream(self, docs: Iterator[Document], buffer_size: int = 10000) -> Iterator[Document]:
        """Globally deduplicate an input stream using bounded input batches.

        The iterator is consumed incrementally, so the complete input corpus is
        never materialized. Output is intentionally delayed until the input is
        exhausted because a later, higher-weight duplicate can replace an earlier
        representative. Only representatives and their embeddings are retained.
        """
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")

        if not self._ensure_embedding_or_fallback():
            yield from self._fallback_stream(docs, buffer_size)
            return

        reps: list[Document] = []
        rep_vecs: list[np.ndarray] = []
        winners: list[Document] = []
        total = 0
        buffer: list[Document] = []

        def process(batch: list[Document]) -> None:
            nonlocal total
            if not batch:
                return
            total += len(batch)
            matrix = self._embed([d.text for d in batch])
            for doc, vec in zip(batch, matrix, strict=True):
                match = None
                if rep_vecs:
                    mat = np.asarray(rep_vecs, dtype=np.float32)
                    idx = self._build_index(mat) if FAISS_AVAILABLE else None
                    if idx is not None:
                        sims, ids = idx.search(vec.reshape(1, -1), min(5, len(rep_vecs)))
                        pairs = zip(sims[0], ids[0], strict=True)
                    else:
                        sims, ids = _cosine_brute(mat, vec, min(5, len(rep_vecs)))
                        pairs = zip(sims, ids, strict=True)
                    for sim, rid in pairs:
                        rid = int(rid)
                        if rid >= 0 and float(sim) >= self._threshold:
                            match = rid
                            break
                if match is None:
                    reps.append(doc)
                    rep_vecs.append(vec)
                    winners.append(doc)
                elif doc.final_weight > reps[match].final_weight:
                    old = reps[match]
                    reps[match] = doc
                    rep_vecs[match] = vec
                    winners[winners.index(old)] = doc

        for doc in docs:
            buffer.append(doc)
            if len(buffer) >= buffer_size:
                process(buffer)
                buffer.clear()
        if buffer:
            process(buffer)

        self.stats["total"] += total
        self.stats["kept"] += len(winners)
        self.stats["dropped"] += total - len(winners)
        log.info("Semantic dedup stream: %d → %d (-%d)", total, len(winners), total - len(winners))
        yield from winners

    def print_stats(self):
        s = self.stats
        t = max(s["total"], 1)
        log.info("SemanticDedup | total=%s kept=%s dropped=%s (%.1f%%)", s["total"], s["kept"], s["dropped"], s["dropped"] / t * 100)
