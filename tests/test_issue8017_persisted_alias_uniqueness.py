"""Persisted-alias uniqueness regression tests for #8017.

A ``custom_providers[]`` row has TWO identities: the catalog slug
(``custom:<slug>``) is the WebUI routing identity, while the hermes-agent
resolves a PERSISTED provider by the row's own ``name`` / ``provider_key``.
A generated routing slug — the fallback minted when a name slugifies to ""
(e.g. a pure-CJK name) — means nothing to the Agent, so at save time it is
translated to the owning row's name.

Proving the SLUG is unique is not enough. The value being persisted is the
row's NAME, and the Agent resolves a persisted name by FIRST match over the
loaded rows. Two rows can mint DISTINCT generated slugs (the endpoint
fallback embeds each row's own host/port and name hash) while sharing one
name: the slug check passes, the save succeeds, and the main runtime then
resolves the FIRST row's endpoint and key — a credential handed to an
endpoint the user never selected.

The Agent also accepts the supported ``providers:`` map, whose enabled
key/name aliases take precedence over the legacy ``custom_providers`` list,
so a unique legacy name is not necessarily a unique runtime alias.

These tests drive the REAL setters against a real config.yaml on disk and
assert refusal (with a byte-identical file) whenever the name that would be
persisted does not resolve uniquely to the selected row.
"""

from __future__ import annotations

import contextlib

import pytest
import yaml

import api.config as config

DUP_NAME = "晨光"
URL_FIRST = "http://127.0.0.1:8318/v1"
URL_SECOND = "http://127.0.0.1:8317/v1"
KEY_FIRST = "test-skSHADOW"
KEY_SECOND = "test-skENTRY"
MODEL = "chat-model"


@pytest.fixture(autouse=True)
def _isolate_models_cache():
    try:
        config.invalidate_models_cache()
    except Exception:
        pass
    yield
    try:
        config.invalidate_models_cache()
    except Exception:
        pass


def _slug(entry: dict) -> str:
    return config._custom_provider_slug_for_entry(entry)


def _dup_entries():
    """One name, two endpoints/keys — distinct generated slugs, one alias."""
    return [
        {"name": DUP_NAME, "base_url": URL_FIRST, "api_key": KEY_FIRST, "model": MODEL},
        {"name": DUP_NAME, "base_url": URL_SECOND, "api_key": KEY_SECOND, "model": MODEL},
    ]


@contextlib.contextmanager
def _disk_cfg(monkeypatch, tmp_path, config_data, resolved=None):
    """Write *config_data* to a real config.yaml and point the setters at it."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_data, allow_unicode=True), encoding="utf-8"
    )
    monkeypatch.setattr(config, "_get_config_path", lambda: config_path)
    monkeypatch.setattr(config, "reload_config", lambda: None)
    monkeypatch.setattr(config, "invalidate_models_cache", lambda: None)
    if resolved is not None:
        slug, url = resolved
        monkeypatch.setattr(
            config, "resolve_model_provider", lambda model: (model, slug, url)
        )
    yield config_path


def _main_config(entries, providers=None):
    cfg = {
        "model": {"default": MODEL, "provider": "custom"},
        "custom_providers": entries,
    }
    if providers is not None:
        cfg["providers"] = providers
    return cfg


# ── the legacy custom_providers list ────────────────────────────────────────


def test_duplicate_names_refuse_the_main_save(monkeypatch, tmp_path):
    """Two rows sharing one name must not persist a name the Agent resolves
    to the WRONG row's endpoint and key."""
    entries = _dup_entries()
    selected = entries[1]
    with _disk_cfg(
        monkeypatch,
        tmp_path,
        _main_config(entries),
        resolved=(_slug(selected), selected["base_url"]),
    ) as config_path:
        before = config_path.read_bytes()
        with pytest.raises(config.AmbiguousCustomProviderError):
            config.set_hermes_default_model(MODEL, provider=_slug(selected))
        assert config_path.read_bytes() == before, "a refused save must not touch the file"


def test_duplicate_names_refuse_the_auxiliary_save(monkeypatch, tmp_path):
    """The auxiliary slot resolves its key by first match too."""
    entries = _dup_entries()
    selected = entries[1]
    with _disk_cfg(
        monkeypatch, tmp_path, _main_config(entries), resolved=(_slug(selected), selected["base_url"])
    ) as config_path:
        before = config_path.read_bytes()
        with pytest.raises(config.AmbiguousCustomProviderError):
            config.set_auxiliary_model("compression", _slug(selected), MODEL)
        assert config_path.read_bytes() == before, "a refused save must not touch the file"


def test_a_unique_cjk_row_persists_its_name(monkeypatch, tmp_path):
    """The single-row case must still save: the generated slug becomes the
    row's name, and the Agent then resolves that name to this row."""
    entry = {
        "name": "晨光鑫遇专用",
        "base_url": "http://127.0.0.1:8317/v1",
        "api_key": "test-skENTRY",
        "model": MODEL,
    }
    slug = _slug(entry)
    with _disk_cfg(
        monkeypatch, tmp_path, _main_config([entry]), resolved=(slug, entry["base_url"])
    ) as config_path:
        config.set_hermes_default_model(MODEL, provider=slug)
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["model"]["provider"] == "晨光鑫遇专用"


def test_an_ascii_name_persists_unchanged(monkeypatch, tmp_path):
    """A name-derived slug IS the Agent identity — it must persist verbatim."""
    entry = {
        "name": "cgxy-cpa",
        "base_url": "http://127.0.0.1:8317/v1",
        "api_key": "test-skENTRY",
        "model": MODEL,
    }
    with _disk_cfg(
        monkeypatch, tmp_path, _main_config([entry]), resolved=("custom:cgxy-cpa", entry["base_url"])
    ) as config_path:
        config.set_hermes_default_model(MODEL, provider="custom:cgxy-cpa")
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["model"]["provider"] == "custom:cgxy-cpa"


# ── the raw providers: map takes precedence over the legacy list ─────────────


def test_a_raw_provider_shadowing_the_name_refuses_the_save(monkeypatch, tmp_path):
    """An enabled ``providers:`` record answering to the same alias would
    shadow the legacy row the Agent resolves first."""
    entry = {
        "name": "晨光",
        "base_url": "http://127.0.0.1:8317/v1",
        "api_key": "test-skENTRY",
        "model": MODEL,
    }
    slug = _slug(entry)
    providers = {
        "晨光": {
            "name": "星野",
            "api": "http://127.0.0.1:8318/v1",
            "api_key": "test-skSHADOW",
            "default_model": MODEL,
        }
    }
    with _disk_cfg(
        monkeypatch,
        tmp_path,
        _main_config([entry], providers=providers),
        resolved=(slug, entry["base_url"]),
    ) as config_path:
        before = config_path.read_bytes()
        with pytest.raises(config.AmbiguousCustomProviderError):
            config.set_hermes_default_model(MODEL, provider=slug)
        assert config_path.read_bytes() == before, "a refused save must not touch the file"


def test_a_disabled_raw_provider_does_not_block_the_save(monkeypatch, tmp_path):
    """A DISABLED raw record cannot shadow anything — the save must proceed."""
    entry = {
        "name": "晨光",
        "base_url": "http://127.0.0.1:8317/v1",
        "api_key": "test-skENTRY",
        "model": MODEL,
    }
    slug = _slug(entry)
    providers = {
        "晨光": {
            "name": "星野",
            "api": "http://127.0.0.1:8318/v1",
            "api_key": "test-skSHADOW",
            "enabled": False,
            "default_model": MODEL,
        }
    }
    with _disk_cfg(
        monkeypatch,
        tmp_path,
        _main_config([entry], providers=providers),
        resolved=(slug, entry["base_url"]),
    ) as config_path:
        config.set_hermes_default_model(MODEL, provider=slug)
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["model"]["provider"] == "晨光"


# ── the translation itself ───────────────────────────────────────────────────


def test_persistence_translation_leaves_a_builtin_provider_alone():
    assert config._agent_provider_identity_for_persistence("openai", {}) == "openai"


def test_persistence_translation_returns_an_unowned_slug_unchanged():
    assert config._agent_provider_identity_for_persistence("custom:ghost", {}) == "custom:ghost"


def test_persistence_translation_maps_a_generated_slug_to_the_row_name():
    entry = {"name": "晨光鑫遇专用", "base_url": "http://127.0.0.1:8317/v1"}
    slug = _slug(entry)
    assert slug != "custom:晨光鑫遇专用"
    assert config._agent_provider_identity_for_persistence(slug, {"custom_providers": [entry]}) == "晨光鑫遇专用"
