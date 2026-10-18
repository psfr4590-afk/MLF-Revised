from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.crawler.content_parser import parse_content
from ui.services.log_service import LogService


def test_xml_parser_rejects_external_entities():
    payload = b'''<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>'''
    with pytest.raises(Exception):
        parse_content("payload.xml", "application/xml", payload)


def test_log_service_rejects_arbitrary_file(tmp_path: Path):
    secret = tmp_path / "secret.log"
    secret.write_text("not a command-center log", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        LogService().tail(secret, 10)


def test_log_service_rejects_symlink(tmp_path: Path):
    allowed = Path("datasets")
    allowed.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "target.log"
    target.write_text("secret", encoding="utf-8")
    link = allowed / "security-test.log"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    try:
        with pytest.raises(ValueError):
            LogService().tail(link, 10)
    finally:
        link.unlink(missing_ok=True)



def test_import_source_requires_relative_path_inside_root(tmp_path: Path, monkeypatch):
    import command_center.security as security

    root = tmp_path / "imports"
    root.mkdir()
    allowed = root / "allowed.txt"
    allowed.write_text("safe", encoding="utf-8")
    monkeypatch.setattr(security, "IMPORT_ROOT", root)

    assert security.validate_import_source("allowed.txt") == allowed
    with pytest.raises(ValueError, match="inside"):
        security.validate_import_source(str(tmp_path / "outside.txt"))
    with pytest.raises(ValueError, match="inside"):
        security.validate_import_source("../outside.txt")


def test_import_source_rejects_symlink_inside_root(tmp_path: Path, monkeypatch):
    import command_center.security as security

    root = tmp_path / "imports"
    root.mkdir()
    target = tmp_path / "target.txt"
    target.write_text("outside", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    monkeypatch.setattr(security, "IMPORT_ROOT", root)
    try:
        with pytest.raises(ValueError, match="symlink"):
            security.validate_import_source("link.txt")
    finally:
        link.unlink(missing_ok=True)
