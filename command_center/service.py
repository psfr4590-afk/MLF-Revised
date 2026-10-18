from __future__ import annotations

from .config import load_groups
from .security import IMPORT_ROOT, validate_import_source
from .runner import start_stage
from .secrets import credentials
from .store import store

STAGES = ("crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export")
CREDENTIAL_PRESETS = (
    {"name": "github", "provider": "GitHub", "kind": "API token", "env_var": "GITHUB_TOKEN", "description": "GitHub repository crawling."},
    {"name": "huggingface", "provider": "Hugging Face", "kind": "Hub token", "env_var": "HF_TOKEN", "description": "Gated/private Hugging Face datasets."},
    {"name": "google_api", "provider": "Google", "kind": "API key", "env_var": "GOOGLE_API_KEY", "description": "Google Programmable Search JSON API."},
    {"name": "google_cx", "provider": "Google", "kind": "Search engine ID", "env_var": "GOOGLE_CX", "description": "Google Programmable Search engine identifier."},
)


def init():
    IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
    return store.ensure_seed_datasets()


def add(name, description="", group_id=None):
    return store.create(name, description, group_id=group_id)


def ingest(did, path):
    # HTTP/UI ingestion is deliberately confined to the operator-managed import
    # directory. A localhost API is not a magic security boundary: another local
    # process can still forge requests and otherwise turn this endpoint into an
    # arbitrary-file reader.
    return store.ingest_path(did, validate_import_source(path))


def stage(did, name):
    if name not in STAGES:
        raise ValueError(f"Unknown stage: {name}. Valid stages: {', '.join(STAGES)}")
    return start_stage(did, name)


def status(did=None):
    return store.refresh_pipeline_state(did) if did else [store.refresh_pipeline_state(d["id"]) for d in store.list()]


def groups():
    return load_groups()


def credential_presets():
    configured = {item["name"]: item for item in credentials.list()}
    return [{**preset, "configured": bool(configured.get(preset["name"], {}).get("stored")),
             "environment_set": bool(configured.get(preset["name"], {}).get("environment_set"))}
            for preset in CREDENTIAL_PRESETS]


def credential_list():
    return credentials.list()


def credential_set(name, secret, provider="custom", kind="token", env_var="", description="", identity=""):
    return credentials.set(name, secret, provider, kind, env_var, description, identity)


def credential_delete(name):
    return credentials.delete(name)


def credential_test(name):
    return credentials.test(name)


def crawl_stats(did):
    return store.crawl_stats(did)


def crawl_domains(did):
    return store.crawl_domains(did)


def crawl_log(did, tail=80):
    return store.crawl_log(did, tail)
