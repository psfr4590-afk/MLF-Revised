"""Torch-free numeric safety guards shared by training and evaluation."""
from __future__ import annotations

import math
from typing import Optional


class NonFiniteLossError(RuntimeError):
    """Raised when training or evaluation produces NaN/Inf, which must never be persisted."""


def require_finite(name: str, value: float, step: Optional[int] = None) -> float:
    """Return ``value`` as float or raise ``NonFiniteLossError`` for NaN/Inf."""
    number = float(value)
    if not math.isfinite(number):
        where = f" at step {step}" if step is not None else ""
        raise NonFiniteLossError(
            f"Non-finite {name} ({number}){where}; refusing to continue or write checkpoints. "
            "Lower the learning rate, check the data, or inspect the last good checkpoint."
        )
    return number
