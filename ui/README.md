# Legacy desktop UI

This directory is retained only for compatibility with older local installations and legacy tests. It is **not** the Model Lab operator interface.

The supported operator surface is the localhost browser Command Center:

    python run_command_center.py

The Python package metadata intentionally excludes `ui*` from the distributable package. New features must not be added to this directory. New operator behavior belongs in `command_center/` and its browser surface.
