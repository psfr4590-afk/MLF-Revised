"""Model Lab desktop UI core package."""

try:
    from .application import Application
    _has_app = True
except (ImportError, ModuleNotFoundError):
    Application = None
    _has_app = False

from . import events, navigation, state

__all__ = ["events", "navigation", "state"]
if _has_app:
    __all__.append("Application")
