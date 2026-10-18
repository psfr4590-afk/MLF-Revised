"""Resolve special-token IDs from the tokenizer artifacts actually produced.

Export and verification must never assume ``pad=0, bos=2, eos=3``: the IDs
depend on the configured ``special_tokens`` order. This module is torch-free
so release tooling and tests can use it anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

ROLES = ("pad", "unk", "bos", "eos")


def special_token_ids(tokenizer_dir: str | Path) -> dict[str, int]:
    """Return ``{"pad","unk","bos","eos"} -> id`` for a trained tokenizer directory."""
    tokenizer_dir = Path(tokenizer_dir)
    config_path = tokenizer_dir / "tokenizer_config.json"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    for path in (config_path, tokenizer_path):
        if not path.is_file():
            raise RuntimeError(f"Tokenizer artifact is missing: {path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Tokenizer artifacts are unreadable: {exc}") from exc

    vocab = (tokenizer.get("model") or {}).get("vocab")
    token_ids: dict[str, int] = dict(vocab) if isinstance(vocab, dict) else {}
    for added in tokenizer.get("added_tokens") or []:
        if isinstance(added, dict) and "content" in added and "id" in added:
            token_ids[str(added["content"])] = int(added["id"])

    resolved: dict[str, int] = {}
    for role in ROLES:
        token = config.get(f"{role}_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError(f"tokenizer_config.json does not define {role}_token")
        if token not in token_ids:
            raise RuntimeError(f"{role}_token {token!r} is not present in the tokenizer vocabulary")
        resolved[role] = int(token_ids[token])
    return resolved
