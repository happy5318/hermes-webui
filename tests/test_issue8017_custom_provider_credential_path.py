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
    """Write a real config.yaml and point both modules at it.

    The credential lookups go through ``get_config()``, which serves the
    module's in-memory ``cfg``. Pointing ``_get_config_path`` at a file is
    not enough on its own — the live config has to actually BE that file's
    content — so the helper loads it into ``cfg`` and restores the previous
    state afterwards.
    """
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
    original_cfg = config.cfg
    config._get_config_path = lambda: config_path
    config.reload_config = lambda: None
    config.invalidate_models_cache = lambda: None
    # The live config must BE the file's content: the credential lookups
    # read the module's in-memory cfg, not the path.
    config.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        yield config_path
    finally:
        config._get_config_path = original_path
        config.reload_config = original_reload
        config.invalidate_models_cache = original_cache
        config.cfg = original_cfg


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
    """The credential check must recognise the identity the picker advertises.

    Driven through a real config.yaml: an in-memory ``config.cfg`` patch is
    not an override (``cfg is _cfg_cache`` stays true), so a later config
    read can silently replace it — and a preceding test that reloaded the
    config from disk would leave this test asserting against whatever the
    previous test had left behind.
    """
    with _disk_config([_entry()], tmp_path):
        assert providers.provider_has_usable_credential(f"custom:{PROVIDER_KEY}") is True


def test_provider_api_key_resolves_through_the_advertised_identity(monkeypatch, tmp_path):
    """The API key lookup must resolve the record's own key."""
    with _disk_config([_entry()], tmp_path):
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


def test_the_matcher_keeps_the_historical_raw_name_spellings():
    """The raw-name spellings are the historical contract: a record named
    ``Local Proxy`` is reachable as ``custom:local proxy`` (the raw name keeps
    its space) as well as through the name-derived slug. A matcher that only
    compared the canonical key would drop that spelling and break a working
    credential lookup."""
    record = {"name": "Local Proxy"}
    assert providers._custom_provider_name_matches("custom:local proxy", record) is True
    assert providers._custom_provider_name_matches("local proxy", record) is True
    assert providers._custom_provider_name_matches("custom:local-proxy", record) is True
    # A name that slugifies empty has no raw-name spelling to fall back on.
    assert providers._custom_provider_name_matches("custom:晨光", _entry()) is False


def test_an_unrelated_identity_is_not_matched():
    """A different record's identity must not match this record."""
    entry = _entry()
    assert providers._custom_provider_name_matches("custom:someone-else", entry) is False
