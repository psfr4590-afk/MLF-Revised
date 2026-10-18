from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Iterator

log = logging.getLogger("tokenizer")

try:
    import tokenizers as tokenizers_lib
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    HF_TOKENIZERS_AVAILABLE = False
    tokenizers_lib = None
    log.warning("tokenizers library not installed")


def _bytelevel_alphabet() -> list[str]:
    """Return the complete byte-level alphabet in a stable order."""
    return sorted(ByteLevel.alphabet(), key=ord)


def _bytelevel_alphabet_fingerprint() -> str:
    """Fingerprint the exact Unicode symbols used to represent all 256 bytes."""
    payload = "".join(_bytelevel_alphabet()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _doc_stream(jsonl_path: Path, sample_size: int | None, seed: int = 42) -> Iterator[str]:
    """Yield corpus text using bounded-memory reservoir sampling."""
    rng = random.Random(seed)
    if sample_size is None:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if text:
                    yield text
        return

    reservoir: list[str] = []
    seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except (json.JSONDecodeError, AttributeError):
                continue
            if not text:
                continue
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(text)
            else:
                slot = rng.randrange(seen)
                if slot < sample_size:
                    reservoir[slot] = text

    log.info("Tokenizer training: sampled %d / %d usable documents", len(reservoir), seen)
    rng.shuffle(reservoir)
    yield from reservoir


class BPETokenizerTrainer:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._vocab_size = cfg.get("vocab_size", 32000)
        self._min_freq = cfg.get("min_frequency", 2)
        self._special = cfg.get(
            "special_tokens",
            ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|sep|>", "<|mask|>"],
        )
        self._output_path = Path(cfg.get("output_path", "output/tokenizer")).resolve()
        self._sample_size = cfg.get("sample_size", 500_000) if cfg.get("train_on_sample") else None

    def train(self, corpus_path: str | Path):
        if not HF_TOKENIZERS_AVAILABLE:
            raise RuntimeError("tokenizers library not installed; run: pip install tokenizers")
        corpus_path = Path(corpus_path)
        self._output_path.mkdir(parents=True, exist_ok=True)
        log.info("Training BPE tokenizer | vocab=%s min_freq=%s sample=%s", self._vocab_size, self._min_freq, self._sample_size)

        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        # ByteLevel BPE must see the original Unicode text so its byte mapping stays reversible.
        # NFKC here would transform some byte-mapping symbols and break full-byte coverage.
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        alphabet = _bytelevel_alphabet()
        minimum_vocab_size = len(alphabet) + len(set(self._special))
        if self._vocab_size < minimum_vocab_size:
            raise ValueError(
                f"vocab_size={self._vocab_size} is too small for the complete byte-level "
                f"alphabet ({len(alphabet)} symbols) plus {len(set(self._special))} "
                f"unique special tokens; minimum={minimum_vocab_size}."
            )
        trainer = BpeTrainer(
            vocab_size=self._vocab_size,
            min_frequency=self._min_freq,
            initial_alphabet=alphabet,
            special_tokens=self._special,
            show_progress=True,
        )
        tmp_path = self._output_path / "_train_corpus.txt"
        count = 0
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for text in _doc_stream(corpus_path, self._sample_size):
                    f.write(text.replace("\n", " ") + "\n")
                    count += 1
                    if count % 50_000 == 0:
                        log.info("... %d docs written", count)
            if count == 0:
                raise RuntimeError(f"Tokenizer corpus contains no usable documents: {corpus_path}")

            tokenizer.train([str(tmp_path)], trainer)
            actual_vocab_size = tokenizer.get_vocab_size()
            if actual_vocab_size != self._vocab_size:
                raise RuntimeError(
                    "Tokenizer vocabulary contract failed: "
                    f"requested={self._vocab_size}, actual={actual_vocab_size}. "
                    "Increase corpus diversity/size or adjust tokenizer training parameters."
                )

            special_ids = {}
            for token in self._special:
                token_id = tokenizer.token_to_id(token)
                if token_id is None:
                    raise RuntimeError(f"Tokenizer special-token contract failed: {token!r} is missing from the trained vocabulary")
                special_ids[token] = token_id

            out_json = self._output_path / "tokenizer.json"
            tokenizer.save(str(out_json))
            corpus_sha256 = _sha256_file(corpus_path)
            config = {
                "bos_token": "<|bos|>", "eos_token": "<|eos|>", "unk_token": "<|unk|>",
                "pad_token": "<|pad|>", "sep_token": "<|sep|>", "mask_token": "<|mask|>",
                "model_type": "llama", "tokenizer_class": "PreTrainedTokenizerFast",
                "vocab_size": actual_vocab_size,
                "provenance": {
                    "schema": 1,
                    "trainer": "BPETokenizerTrainer",
                    "tokenizers_version": getattr(tokenizers_lib, "__version__", "unknown"),
                    "corpus_sha256": corpus_sha256,
                    "corpus_path": str(corpus_path),
                    "sample_size": self._sample_size,
                    "seed": 42,
                    "vocab_size_requested": self._vocab_size,
                    "min_frequency": self._min_freq,
                    "special_tokens": list(self._special),
                    "bytelevel_alphabet_size": len(alphabet),
                    "bytelevel_alphabet_sha256": _bytelevel_alphabet_fingerprint(),
                    "bytelevel_alphabet_codepoints": [ord(symbol) for symbol in alphabet],
                },
            }
            (self._output_path / "tokenizer_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            spm = {k: config[k] for k in ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "mask_token")}
            (self._output_path / "special_tokens_map.json").write_text(json.dumps(spm, indent=2), encoding="utf-8")

            required = [self._output_path / n for n in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise RuntimeError("Tokenizer artifact contract failed; missing: " + ", ".join(missing))
            saved_config = json.loads((self._output_path / "tokenizer_config.json").read_text(encoding="utf-8"))
            saved_specials = json.loads((self._output_path / "special_tokens_map.json").read_text(encoding="utf-8"))
            if saved_config.get("vocab_size") != self._vocab_size:
                raise RuntimeError(f"Tokenizer config vocab_size contract failed: expected={self._vocab_size}, actual={saved_config.get('vocab_size')}")
            if not saved_config.get("provenance", {}).get("corpus_sha256"):
                raise RuntimeError("Tokenizer provenance contract failed: corpus_sha256 is missing")
            provenance = saved_config.get("provenance", {})
            if provenance.get("bytelevel_alphabet_size") != 256:
                raise RuntimeError("Tokenizer alphabet contract failed: full 256-symbol byte-level alphabet is not recorded")
            if provenance.get("bytelevel_alphabet_sha256") != _bytelevel_alphabet_fingerprint():
                raise RuntimeError("Tokenizer alphabet fingerprint contract failed")
            for name in ("bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "mask_token"):
                if name not in saved_config or name not in saved_specials:
                    raise RuntimeError(f"Tokenizer special-token artifact contract failed: {name}")
            log.info("Tokenizer artifact contract satisfied | vocab=%d | special_tokens=%d", actual_vocab_size, len(special_ids))
            return tokenizer
        finally:
            tmp_path.unlink(missing_ok=True)

    def encode(self, text: str, tokenizer=None) -> list[int]:
        if tokenizer is None:
            raise ValueError("Pass a trained tokenizer instance")
        return tokenizer.encode(text).ids

    def load(self):
        tok_path = self._output_path / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"No tokenizer at {tok_path}; run train() first")
        if not HF_TOKENIZERS_AVAILABLE:
            raise RuntimeError("tokenizers library not installed")
        tok = Tokenizer.from_file(str(tok_path))
        log.info("Loaded tokenizer from %s (vocab=%d)", tok_path, tok.get_vocab_size())
        return tok


__all__ = ["BPETokenizerTrainer", "HF_TOKENIZERS_AVAILABLE"]
