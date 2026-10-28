"""Credential-path regression tests for a non-ASCII-only custom provider.

A record whose name slugifies to "" (a pure-CJK name) is advertised in the
picker under a GENERATED identity (``custom:<provider_key>`` or the
endpoint/name-hash fallback). The credential lookups in ``api/providers.py``
must recognise that SAME identity, or the provider appears in the picker,
routes its models, and yet reports no usable credential — or cannot have its
key removed through the identity the picker advertised.

This is the gap #8031 leaves open: its ``_custom_provider_name_matches()``
resolves a custom provider by its name-derived slug only, so through the
catalog's own advertised identity the credential lookups miss the record.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import yaml

import api.config as config
import api.providers as providers

CJK_NAME = "晨光鑫遇专用"
PROVIDER_KEY = "cgxy-cpa"
BASE_URL = "http://127.0.0.1:8317/v1"
API_KEY = "sk-cgxy"
MODEL = "shared:model"


@contextlib.contextmanager
def _disk_config(entries, tmp_path):
    """Write a real config.yaml and point both modules at it."""
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"default": MODEL, "provider": f"custom:{PROVIDER_KEY}"},
                "custom_providers": entries,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    original_path = config._get_config_path
    original_reload = config.reload_config
    original_cache = config.invalidate_models_cache
    config._get_config_path = lambda: config_path
    config.reload_config = lambda: None
    config.invalidate_models_cache = lambda: None
    try:
        yield config_path
    finally:
        config._get_config_path = original_path
        config.reload_config = original_reload
        config.invalidate_models_cache = original_cache


def _entry():
    return {
        "name": CJK_NAME,
        "provider_key": PROVIDER_KEY,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "model": MODEL,
    }


def test_the_advertised_identity_is_the_provider_key_slug():
    """The identity the catalog advertises for this record."""
    assert config._custom_provider_slug_for_entry(_entry()) == f"custom:{PROVIDER_KEY}"


def test_provider_has_usable_credential_through_the_advertised_identity(monkeypatch, tmp_path):
    """The credential check must recognise the identity the picker advertises."""
    entry = _entry()
    monkeypatch.setattr(
        config, "get_config", lambda: {"custom_providers": [entry], "model": {"provider": "custom"}}
    )
    monkeypatch.setitem(config.cfg, "custom_providers", [entry])
    assert providers.provider_has_usable_credential(f"custom:{PROVIDER_KEY}") is True


def test_provider_api_key_resolves_through_the_advertised_identity(monkeypatch):
    """The API key lookup must resolve the record's own key."""
    entry = _entry()
    monkeypatch.setattr(
        config, "get_config", lambda: {"custom_providers": [entry], "model": {"provider": "custom"}}
    )
    monkeypatch.setitem(config.cfg, "custom_providers", [entry])
    assert providers._get_provider_api_key(f"custom:{PROVIDER_KEY}") == API_KEY


def test_the_key_can_be_removed_through_the_advertised_identity(monkeypatch, tmp_path):
    """Clearing the key through the advertised identity must actually remove
    it from the custom_providers[] row."""
    with _disk_config([_entry()], tmp_path) as config_path:
        providers._clean_provider_key_from_config(f"custom:{PROVIDER_KEY}")
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["custom_providers"][0].get("api_key") is None


def test_the_matcher_recognises_the_advertised_identity():
    """The record-aware matcher must accept the advertised identity."""
    entry = _entry()
    assert providers._custom_provider_name_matches(f"custom:{PROVIDER_KEY}", entry) is True


def test_an_unrelated_identity_is_not_matched():
    """A different record's identity must not match this record."""
    entry = _entry()
    assert providers._custom_provider_name_matches("custom:someone-else", entry) is False
