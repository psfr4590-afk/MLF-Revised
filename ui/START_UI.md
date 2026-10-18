# Model Lab Command Center

The project has exactly one operator interface: the localhost browser Command Center.

`launch.py` and this compatibility path both delegate to `run_command_center.py`. The old Tkinter implementation is no longer an operator surface and is not part of the release contract.

Canonical launch:

```powershell
python .\\launch.py
```

The Command Center reads operational state from the authoritative SQLite experiment store through the FastAPI service. It does not maintain a parallel UI state store.
