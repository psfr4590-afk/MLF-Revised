#!/usr/bin/env python3
"""Canonical CLI entry point for the Model Lab end-to-end training flow."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.version import get_version  # noqa: E402

STAGES = ["crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export"]
VERSION = get_version()
DEFAULT_CONFIG = "config/pipeline_config.yaml"
DATASET_SESSION_CONFIG = "config/pipeline_config.dataset.yaml"


def _load_cli_dataset_groups(config_path: Path) -> list[dict]:
    """Load dataset-group metadata without constructing a pipeline."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    configured = config.get("crawl", {}).get("dataset_groups_file", "config/dataset_groups.yaml")
    groups_path = Path(configured)
    if not groups_path.is_absolute():
        groups_path = ROOT / groups_path
    if not groups_path.is_file():
        return [{"id": "default", "name": "default"}]
    payload = yaml.safe_load(groups_path.read_text(encoding="utf-8")) or {}
    return list(payload.get("dataset_groups", []))


def _parse_requested_stages(value: str) -> tuple[str, list[str] | None]:
    """Validate a stage selection before constructing the runtime pipeline."""
    requested = value.strip().lower()
    if requested == "all":
        return requested, None
    stages = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown or not stages:
        return requested, None
    return requested, stages


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=f"Model Lab {VERSION}: local-first data and pretraining pipeline.",
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py --doctor\n"
            "  python run_pipeline.py --list-stages\n"
            "  python run_pipeline.py --list-groups\n"
            "  python run_pipeline.py --no-resume\n"
            "  mlab --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main() -> int:
    parser = _build_parser()
    parser.add_argument("--version", action="version", version=f"Model Lab {VERSION}")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="pipeline config (default: %(default)s)")
    parser.add_argument("--stages", default="all", help="comma-separated stages or 'all' (default: all; uses config stage enablement)")
    parser.add_argument("--no-resume", action="store_true", help="disable all artifact and checkpoint resume for this run")
    parser.add_argument("--dataset-group", default=None, help="dataset group ID to run")
    parser.add_argument("--dataset-id", type=int, default=None, help="numeric dataset ID override")
    parser.add_argument("--doctor", action="store_true", help="check project/runtime readiness without running stages")
    parser.add_argument(
        "--hardware-report",
        action="store_true",
        help="include detected hardware and conservative training guidance (with --doctor)",
    )
    parser.add_argument("--list-groups", action="store_true", help="list configured dataset groups without starting the pipeline")
    parser.add_argument("--list-stages", action="store_true", help="list pipeline stages without starting the pipeline")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="logging verbosity")
    parser.add_argument("--web", action="store_true", help="start the backend command center instead of the pipeline")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.hardware_report and not args.doctor:
        print("--hardware-report requires --doctor")
        return 1

    if args.doctor:
        from pipeline.doctor import run_doctor
        ok, checks = run_doctor(ROOT)
        print("Model Lab Doctor")
        for check in checks:
            marker = "PASS" if check["ok"] else ("WARN" if not check["required"] else "FAIL")
            print(f"[{marker}] {check['name']}: {check['detail']}")
        if args.hardware_report:
            from pipeline.model_sizer import profile_hardware, recommend_training_profile
            hardware = profile_hardware()
            profile = recommend_training_profile(hardware)
            print("Hardware Report")
            print(json.dumps({"hardware": hardware.to_dict(), "recommendation": profile.to_dict()}, indent=2, sort_keys=True))
        return 0 if ok else 2

    if args.list_stages:
        descriptions = {
            "crawl": "Crawl configured sources",
            "clean": "Normalize, filter, and exact-deduplicate",
            "dedup": "Semantic near-dedup",
            "weight": "Apply source/content weighting",
            "tokenize": "Train tokenizer",
            "shard": "Write binary training shards",
            "train": "Train the configured model",
            "export": "Export HF/GGUF/Ollama artifacts",
        }
        print("Available stages:")
        for stage in STAGES:
            print(f"  {stage:<12} {descriptions[stage]}")
        return 0

    os.chdir(ROOT)
    if args.web:
        try:
            import uvicorn
            from command_center.config import load_pipeline_config
            ccfg = load_pipeline_config().get("command_center", {})
            uvicorn.run(
                "command_center.web:app",
                host=ccfg.get("host", "127.0.0.1"),
                port=int(ccfg.get("port", 8000)),
                reload=False,
            )
            return 0
        except ImportError as exc:
            print(f"Command center dependencies are missing: {exc}")
            return 1

    config = Path(args.config)
    if not config.is_absolute():
        config = ROOT / config
    if not config.is_file():
        print(f"Config not found: {config}")
        return 1

    if args.dataset_id is not None and args.config == DEFAULT_CONFIG:
        session_config = ROOT / DATASET_SESSION_CONFIG
        if not session_config.is_file():
            print(f"Dataset session config not found: {session_config}")
            return 1
        config = session_config

    if args.list_groups:
        print("Configured dataset groups:")
        for group in _load_cli_dataset_groups(config):
            print(f"  {group.get('id', ''):<32} {group.get('name', '')}")
        return 0

    if args.dataset_id is not None:
        from pipeline.dataset_identity import validate_session_identity
        try:
            validate_session_identity(ROOT, args.dataset_id)
        except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"Dataset identity validation failed: {exc}")
            return 1

    requested, stages = _parse_requested_stages(args.stages)
    if stages is None and requested != "all":
        unknown = [item.strip() for item in requested.split(",") if item.strip() and item.strip() not in STAGES]
        detail = unknown if unknown else [requested or "<empty>"]
        print(f"Unknown stages: {detail}. Valid: {STAGES}")
        return 1

    from pipeline.orchestrator import Pipeline
    pipeline = Pipeline(str(config), dataset_id=args.dataset_id, resume=not args.no_resume)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        if requested == "all":
            pipeline.run("all", dataset_group=args.dataset_group)
        else:
            pipeline.run(",".join(stages or []), dataset_group=args.dataset_group)
    except KeyboardInterrupt:
        print("\nInterrupted. Atomic artifacts remain intact; rerun with resume enabled.")
        return 130
    except Exception as exc:
        print(f"\nPIPELINE FAILED: {type(exc).__name__}: {exc}")
        return 2
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
