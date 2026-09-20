"""Regression tests for #7412: provider removal must clean up the
credential pool entry and mark the env source as suppressed.

Before this fix, the WebUI's ``remove_provider_key()`` only mutated
``.env`` and ``config.yaml``. The env-seeded row in
``~/.hermes/auth.json`` -> ``credential_pool.<provider>`` survived the
removal, so the dead provider stayed visible across a full server
restart and live ``/v1/models`` fetches pointed at a deleted env var.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


class _FakeEntry:
    """Minimal PooledCredential stand-in for tests.

    The real PooledCredential lives in ``agent.credential_pool`` which
    is not on the test process's import path. We only need ``source``,
    ``label``, and equality-by-identity for ``list.index()`` to work.
    """

    def __init__(self, source: str, label: str = "fake"):
        self.source = source
        self.label = label

    def __repr__(self) -> str:
        return f"_FakeEntry(source={self.source!r}, label={self.label!r})"


class _FakePool:
    def __init__(self, entries: list[_FakeEntry] | None = None):
        self._entries = list(entries or [])
        self.removed_indices: list[int] = []

    def entries(self):
        return list(self._entries)

    def remove_index(self, idx: int):
        if idx < 0 or idx >= len(self._entries):
            return None
        removed = self._entries.pop(idx)
        self.removed_indices.append(idx)
        return removed


class _FakeAuthStore:
    def __init__(self):
        self.suppressed: dict[str, list[str]] = {}


@pytest.fixture
def providers_module(monkeypatch, tmp_path):
    """Import ``api.providers`` with fake Agent-side dependencies so
    the test can drive ``remove_provider_key()`` end-to-end without
    touching the real credential pool.

    We patch the public ``load_pool`` / ``suppress_credential_source``
    / ``invalidate_credential_pool_cache`` symbols in-place on the
    real modules so the rest of ``api.providers`` (which imports many
    other names from ``api.config``) keeps working.
    """
    cp_pools: dict[str, _FakePool] = {}

    def _fake_load_pool(provider: str) -> _FakePool:
        return cp_pools.setdefault(provider, _FakePool())

    auth_store = _FakeAuthStore()

    def _fake_suppress(provider: str, source: str) -> None:
        auth_store.suppressed.setdefault(provider, []).append(source)

    invalidated: list[str] = []

    def _fake_invalidate(provider: str) -> None:
        invalidated.append(provider)

    # Inject the stubs into the real ``agent.credential_pool`` /
    # ``hermes_cli.auth`` modules so the imports inside the helper
    # resolve. Use monkeypatch.setattr on the real module to ensure
    # the function reference is captured at call time (the helper
    # imports at call time, not at module load, so this is
    # sufficient).
    import agent.credential_pool as _cp  # type: ignore
    import hermes_cli.auth as _auth  # type: ignore
    import api.config as _config  # type: ignore
    monkeypatch.setattr(_cp, "load_pool", _fake_load_pool)
    monkeypatch.setattr(_auth, "suppress_credential_source", _fake_suppress)
    monkeypatch.setattr(_config, "invalidate_credential_pool_cache", _fake_invalidate)

    # Reload api.providers so the helpers pick up the patched
    # attribute references on the next import. (The patched names
    # are only read at call time, but the api.providers module
    # itself caches some imports from api.config -- the simplest
    # safe path is to reload it so the new attribute is visible.)
    if "api.providers" in sys.modules:
        del sys.modules["api.providers"]
    mod = importlib.import_module("api.providers")
    return mod, cp_pools, auth_store, invalidated


# ── Happy path: env-sourced entry is dropped from the pool ──────────


def test_env_sourced_entry_is_removed_from_pool(providers_module):
    mod, cp_pools, auth_store, invalidated = providers_module
    env_entry = _FakeEntry("env:OPENROUTER_API_KEY", label="or-1")
    other = _FakeEntry("manual", label="manual-1")
    cp_pools["openrouter"] = _FakePool([env_entry, other])

    # Drive the pool-purge step directly (the public route is
    # ``remove_provider_key`` which also mutates .env/config.yaml --
    # exercised end-to-end below).
    mod._purge_provider_from_credential_pool("openrouter")

    pool = cp_pools["openrouter"]
    sources = [e.source for e in pool.entries()]
    assert "env:OPENROUTER_API_KEY" not in sources
    assert "manual" in sources  # non-env sources are left alone
    # The cache must be invalidated so the next read goes through
    # load_pool() and sees the cleaned state.
    assert invalidated == ["openrouter"]
    # The env source is marked suppressed.
    assert "env:OPENROUTER_API_KEY" in auth_store.suppressed.get("openrouter", [])


def test_multiple_env_sourced_entries_are_all_purged(providers_module):
    mod, cp_pools, auth_store, _ = providers_module
    env_a = _FakeEntry("env:OPENROUTER_API_KEY", label="or-1")
    env_b = _FakeEntry("env:OPENAI_API_KEY", label="oa-1")
    cp_pools["openrouter"] = _FakePool([env_a, env_b])

    mod._purge_provider_from_credential_pool("openrouter")

    pool = cp_pools["openrouter"]
    assert pool.entries() == []
    suppressed = set(auth_store.suppressed.get("openrouter", []))
    assert "env:OPENROUTER_API_KEY" in suppressed
    assert "env:OPENAI_API_KEY" in suppressed


def test_non_env_sourced_entries_are_left_alone(providers_module):
    mod, cp_pools, auth_store, _ = providers_module
    manual = _FakeEntry("manual", label="m-1")
    oauth = _FakeEntry("gh_cli", label="gh-1")
    cp_pools["openrouter"] = _FakePool([manual, oauth])

    mod._purge_provider_from_credential_pool("openrouter")

    # Neither manual nor gh_cli is env-sourced, so neither is purged
    # and neither is added to suppressed_sources.
    pool = cp_pools["openrouter"]
    assert pool.entries() == [manual, oauth]
    assert "openrouter" not in auth_store.suppressed


def test_no_env_sourced_entries_is_a_noop(providers_module):
    """A provider with no env-sourced pool entries should not be
    invalidated, not be suppressed, and not raise. The fix is
    fail-closed so a stray ``load_pool`` call cannot turn a clean
    pool into a false-positive suppression.
    """
    mod, cp_pools, auth_store, invalidated = providers_module
    cp_pools["openrouter"] = _FakePool([_FakeEntry("manual")])

    mod._purge_provider_from_credential_pool("openrouter")

    assert invalidated == []
    assert "openrouter" not in auth_store.suppressed


# ── Fail-closed: missing dependencies do not crash the removal ──────


def test_missing_agent_credential_pool_module_is_swallowed(monkeypatch):
    """If ``agent.credential_pool`` cannot be imported (e.g. a
    stripped-down test env) the purge step must be a no-op rather
    than roll back the .env/config.yaml half of the removal the
    caller has already done.
    """
    # Patch ``load_pool`` to raise ImportError so the call-time
    # import inside the helper bubbles up. The helper must swallow
    # the failure and return cleanly.
    import agent.credential_pool as _cp
    def _boom():
        raise ImportError("simulated missing dependency")
    monkeypatch.setattr(_cp, "load_pool", _boom)

    # Reload api.providers so the patched attribute is visible.
    if "api.providers" in sys.modules:
        del sys.modules["api.providers"]
    mod = importlib.import_module("api.providers")
    # Must not raise even though load_pool() would.
    mod._purge_provider_from_credential_pool("openrouter")
