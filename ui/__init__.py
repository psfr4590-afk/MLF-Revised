"""Optional desktop control surface for Model Lab."""

try:
    from .core.application import Application
    __all__ = ["Application"]
except (ImportError, ModuleNotFoundError):
    __all__ = []
