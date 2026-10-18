"""Tiny UI event bus with non-fatal callback isolation."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

_subscribers: dict[str, list[Callable[[Any], None]]] = defaultdict(list)
NAV_CHANGE = "nav.change"
REFRESH = "refresh"
STATUS = "status"
log = logging.getLogger(__name__)


def subscribe(event: str, callback: Callable[[Any], None]) -> None:
    _subscribers[event].append(callback)


def unsubscribe(event: str, callback: Callable[[Any], None]) -> None:
    try:
        _subscribers[event].remove(callback)
    except ValueError:
        pass


def publish(event: str, data: Any = None) -> None:
    """Notify subscribers while keeping one broken callback from stopping the bus."""
    for callback in tuple(_subscribers[event]):
        try:
            callback(data)
        except Exception:
            log.exception("UI event callback failed: event=%s callback=%r", event, callback)
