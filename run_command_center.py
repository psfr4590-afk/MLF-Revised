from __future__ import annotations

import argparse
import threading
import time
import urllib.request
import webbrowser

import uvicorn

from command_center.app import app
from command_center.config import load_pipeline_config


def _server_config() -> tuple[str, int]:
    config = load_pipeline_config().get("command_center", {})
    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 8000))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Command center host must remain loopback-only")
    if not 1 <= port <= 65535:
        raise ValueError("Command center port must be between 1 and 65535")
    return host, port


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Model Lab local command center")
    p.add_argument("--no-browser", action="store_true", help="do not open the localhost command center in a browser")
    args = p.parse_args()
    host, port = _server_config()
    base_url = f"http://{host if host != '::1' else '[::1]'}:{port}"

    if not args.no_browser:
        def _open_browser() -> None:
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"{base_url}/api/system", timeout=0.5):
                        webbrowser.open(base_url)
                        return
                except Exception:
                    time.sleep(0.25)

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
