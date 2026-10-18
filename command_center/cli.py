"""Compatibility CLI for the local command center."""
from __future__ import annotations

import argparse
import json

from .service import stage, status
from .runner import stop


def main(argv=None):
    parser = argparse.ArgumentParser(description="M²S Model Training Pipeline command center")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("datasets", help="List dataset state")
    run = sub.add_parser("run", help="Run one pipeline stage")
    run.add_argument("dataset_id", type=int)
    run.add_argument("stage")
    halt = sub.add_parser("stop", help="Stop a dataset pipeline")
    halt.add_argument("dataset_id", type=int)
    args = parser.parse_args(argv)
    if args.cmd == "datasets":
        print(json.dumps(status(), indent=2, default=str))
    elif args.cmd == "run":
        print(json.dumps(stage(args.dataset_id, args.stage), indent=2, default=str))
    elif args.cmd == "stop":
        print(json.dumps({"stopped": stop(args.dataset_id)}, indent=2, default=str))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
