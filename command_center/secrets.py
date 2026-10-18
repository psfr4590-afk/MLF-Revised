from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from threading import RLock

from cryptography.fernet import Fernet

from .config import ROOT

LOCK = RLock()
STORE_PATH = ROOT / ".runtime" / "credentials.json"
ALLOWED_ENV_VARS = frozenset({
    "GITHUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "ARXIV_API_KEY",
    "GOOGLE_API_KEY", "GOOGLE_CSE_ID", "GOOGLE_CX",
})
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


def _key():
    key = os.environ.get("PIPELINE_CREDENTIAL_KEY", "")
    if not key:
        raise RuntimeError("PIPELINE_CREDENTIAL_KEY is required to access the credential store")
    return Fernet(key.encode())


def validate_env_var(env_var: str) -> str:
    value = str(env_var or "").strip()
    if not value:
        return ""
    if not _ENV_NAME.fullmatch(value) or value not in ALLOWED_ENV_VARS:
        raise ValueError("Unsupported credential environment variable")
    return value


class CredentialStore:
    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if not self.path.exists(): return {"version": 1, "credentials": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def set(self, name, secret, provider="custom", kind="token", env_var="", description="", identity=""):
        env_var = validate_env_var(env_var)
        if not str(name).strip(): raise ValueError("Credential name is required")
        if not isinstance(secret, str): raise ValueError("Credential secret must be text")
        with LOCK:
            data = self._load(); fernet = _key()
            data["credentials"][name] = {"secret": base64.b64encode(fernet.encrypt(secret.encode())).decode(), "provider": provider, "type": kind, "env_var": env_var, "description": description, "identity": identity}
            self._save(data)
            return {"name": name, "provider": provider, "type": kind, "env_var": env_var, "description": description, "identity": identity, "stored": True}

    def reveal(self, name):
        data = self._load()
        if name not in data.get("credentials", {}): raise KeyError(name)
        return _key().decrypt(base64.b64decode(data["credentials"][name]["secret"])).decode()

    def list(self):
        data = self._load(); out = []
        for name, item in sorted(data.get("credentials", {}).items()):
            env_var = validate_env_var(item.get("env_var", ""))
            out.append({"name": name, "provider": item.get("provider", "custom"), "type": item.get("type", "token"), "env_var": env_var, "description": item.get("description", ""), "identity": item.get("identity", ""), "stored": True, "environment_set": bool(env_var and os.environ.get(env_var))})
        return out

    def environment(self):
        env = {}
        for name, item in self._load().get("credentials", {}).items():
            env_var = validate_env_var(item.get("env_var", ""))
            if env_var: env[env_var] = self.reveal(name)
        return env

    def delete(self, name):
        with LOCK:
            data = self._load()
            if name not in data.get("credentials", {}): return False
            del data["credentials"][name]; self._save(data); return True

    def test(self, name):
        value = self.reveal(name); return {"ok": bool(value), "name": name}


credentials = CredentialStore()
