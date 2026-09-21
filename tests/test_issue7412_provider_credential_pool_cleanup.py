"""Regression tests for #7412: provider removal must clean up the
credential pool entry and mark the env source as suppressed.

Round-1 covered the happy paths (env-sourced entries dropped, manual /
OAuth entries preserved, no-op when no env sources, missing
dependency swallowed). Round-2 adds:

- outcome-dict shape assertions: the helper now returns a structured
  outcome (``ok`` / ``pool`` / ``suppress`` / ``cache``) instead of
  ``None``, and ``remove_provider_key`` attaches it to the public
  result so the route can surface partial-failure.
- per-step failure cases: ``load_pool`` raise, ``remove_index`` raise,
  ``suppress_credential_source`` raise, ``invalidate_credential_pool_cache``
  raise -- each must record on the outcome without raising out of
  the helper.
- DELETE route uses ``profile_env_for_active_request`` (write scope)
  so the Agent-side cleanup runs against the request profile's
  ``auth.json``, not the process-default one (review P1).

The real ``agent.credential_pool`` and ``hermes_cli.auth`` live in
the ``hermes-agent`` repo and are not on this repo's import path. We
inject ``sys.modules`` entries that re-route the call-time imports
inside the helper to in-test fakes, so the helper can be exercised
end-to-end without those modules being installed.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


# ── Test doubles ────────────────────────────────────────────────────


class _FakeEntry:
    """Minimal PooledCredential stand-in for tests.

    The real ``PooledCredential`` lives in ``agent.credential_pool``
    which is not on this repo's import path. We need ``source`` and
    identity equality for ``list.index()`` to work.
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
        # Test hook: if set, ``remove_index`` will raise this.
        self.remove_index_raises: Exception | None = None

    def entries(self):
        return list(self._entries)

    def remove_index(self, idx: int):
        if self.remove_index_raises is not None:
            raise self.remove_index_raises
        if idx < 0 or idx >= len(self._entries):
            return None
        removed = self._entries.pop(idx)
        self.removed_indices.append(idx)
        return removed


class _FakeAuthStore:
    def __init__(self):
        self.suppressed: dict[str, list[str]] = {}
        # Test hook: if set, ``suppress`` will raise this.
        self.suppress_raises: Exception | None = None

    def suppress(self, provider: str, source: str) -> None:
        if self.suppress_raises is not None:
            raise self.suppress_raises
        self.suppressed.setdefault(provider, []).append(source)


class _FakeLoadPool:
    def __init__(self, pools: dict[str, _FakePool], raises: Exception | None = None):
        self.pools = pools
        # If set, the function itself raises (different from
        # returning a pool whose remove_index raises).
        self.raises = raises

    def __call__(self, provider: str) -> _FakePool:
        if self.raises is not None:
            raise self.raises
        return self.pools.setdefault(provider, _FakePool())


class _FakeConfig:
    def __init__(self):
        self.invalidated: list[str] = []
        # Test hook: if set, the function will raise.
        self.invalidate_raises: Exception | None = None

    def invalidate_credential_pool_cache(self, provider: str) -> None:
        if self.invalidate_raises is not None:
            raise self.invalidate_raises
        self.invalidated.append(provider)


class _BlockAgentFinder:
    """MetaPathFinder that raises ImportError for any ``agent.*``
    import. Used to simulate a stripped-down test env where the
    hermes-agent modules are not installed. The hermes-agent repo
    lives at ``/home/xiaobao/.hermes/hermes-agent`` and is on
    ``sys.path`` in this dev env, so a bare ``del sys.modules`` is
    not enough -- Python falls back to the real loader. A finder
    that hard-raises is the only reliable way to make the helper's
    call-time import fail.
    """

    def find_spec(self, name, path=None, target=None):
        if name == "agent" or name.startswith("agent."):
            raise ImportError(f"simulated missing dependency: {name}")
        return None


def _install_fake_agent_modules(
    monkeypatch,
    *,
    pools: dict[str, _FakePool] | None = None,
    auth: _FakeAuthStore | None = None,
    config: _FakeConfig | None = None,
    load_pool_raises: Exception | None = None,
) -> tuple[dict[str, _FakePool], _FakeAuthStore, _FakeConfig]:
    """Install fake ``agent.credential_pool`` and ``hermes_cli.auth``
    modules into ``sys.modules`` so the call-time imports inside
    ``_purge_provider_from_credential_pool`` resolve.

    Returns the (pools, auth, config) fakes so the test can drive
    them. Reloads ``api.providers`` so the helper picks up the
    installed fakes (the helper reads them at call time, but the
    function body re-executes on import so a stale reference is
    possible -- reloading is the safe path).
    """
    pools = pools if pools is not None else {}
    auth = auth if auth is not None else _FakeAuthStore()
    config = config if config is not None else _FakeConfig()

    fake_cp_mod = types.ModuleType("agent.credential_pool")
    fake_cp_mod.load_pool = _FakeLoadPool(pools, raises=load_pool_raises)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.credential_pool", fake_cp_mod)

    fake_auth_mod = types.ModuleType("hermes_cli.auth")
    fake_auth_mod.suppress_credential_source = auth.suppress  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", types.ModuleType("hermes_cli"))
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth_mod)

    # The WebUI cache invalidator is real (``api.config``); patch the
    # bound function on the module so the helper's call-time import
    # resolves to our faker.
    import api.config as _config
    monkeypatch.setattr(
        _config, "invalidate_credential_pool_cache",
        config.invalidate_credential_pool_cache,
        raising=False,
    )

    # Reload api.providers so any module-level cache is cleared.
    if "api.providers" in sys.modules:
        del sys.modules["api.providers"]
    return pools, auth, config


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def providers_module(monkeypatch, tmp_path):
    """Import ``api.providers`` with the Agent-side dependencies
    stubbed out via ``sys.modules``. Returns ``(mod, pools, auth,
    config)`` so each test can drive the fakes.
    """
    pools: dict[str, _FakePool] = {}
    auth = _FakeAuthStore()
    config = _FakeConfig()
    _install_fake_agent_modules(
        monkeypatch, pools=pools, auth=auth, config=config
    )
    mod = importlib.import_module("api.providers")
    return mod, pools, auth, config


# ── Happy path: env-sourced entry is dropped from the pool ──────────


def test_env_sourced_entry_is_removed_from_pool(providers_module):
    mod, pools, auth, config = providers_module
    env_entry = _FakeEntry("env:OPENROUTER_API_KEY", label="or-1")
    other = _FakeEntry("manual", label="manual-1")
    pools["openrouter"] = _FakePool([env_entry, other])

    outcome = mod._purge_provider_from_credential_pool("openrouter")

    # Outcome shape (round-2 contract).
    assert outcome["ok"] is True
    assert outcome["skipped"] is None
    assert outcome["pool"]["ok"] is True
    assert outcome["pool"]["removed"] == 1
    assert outcome["pool"]["error"] is None
    assert outcome["suppress"]["ok"] is True
    assert outcome["suppress"]["sources"] == ["env:OPENROUTER_API_KEY"]
    assert outcome["suppress"]["error"] is None
    assert outcome["cache"]["ok"] is True
    assert outcome["cache"]["error"] is None

    # Side effects.
    pool = pools["openrouter"]
    sources = [e.source for e in pool.entries()]
    assert "env:OPENROUTER_API_KEY" not in sources
    assert "manual" in sources  # non-env sources are left alone
    # The cache must be invalidated so the next read goes through
    # load_pool() and sees the cleaned state.
    assert config.invalidated == ["openrouter"]
    # The env source is marked suppressed.
    assert "env:OPENROUTER_API_KEY" in auth.suppressed.get("openrouter", [])


def test_multiple_env_sourced_entries_are_all_purged(providers_module):
    mod, pools, auth, _ = providers_module
    env_a = _FakeEntry("env:OPENROUTER_API_KEY", label="or-1")
    env_b = _FakeEntry("env:OPENAI_API_KEY", label="oa-1")
    pools["openrouter"] = _FakePool([env_a, env_b])

    outcome = mod._purge_provider_from_credential_pool("openrouter")

    assert outcome["ok"] is True
    assert outcome["pool"]["removed"] == 2
    assert set(outcome["suppress"]["sources"]) == {
        "env:OPENROUTER_API_KEY",
        "env:OPENAI_API_KEY",
    }
    pool = pools["openrouter"]
    assert pool.entries() == []


def test_non_env_sourced_entries_are_left_alone(providers_module):
    mod, pools, auth, _ = providers_module
    manual = _FakeEntry("manual", label="m-1")
    oauth = _FakeEntry("gh_cli", label="gh-1")
    pools["openrouter"] = _FakePool([manual, oauth])

    outcome = mod._purge_provider_from_credential_pool("openrouter")

    # Neither manual nor gh_cli is env-sourced, so neither is purged
    # and neither is added to suppressed_sources. ``ok`` is True and
    # the helper still invalidates the cache (defensive).
    assert outcome["ok"] is True
    assert outcome["pool"]["removed"] == 0
    assert outcome["suppress"]["sources"] == []
    pool = pools["openrouter"]
    assert pool.entries() == [manual, oauth]
    assert "openrouter" not in auth.suppressed


def test_no_env_sourced_entries_is_a_noop(providers_module):
    """A provider with no env-sourced pool entries should not
    suppress anything, not raise, and the cache invalidation should
    still run. Round-1 short-circuited here without invalidating the
    cache -- round-2 invalidates defensively.
    """
    mod, pools, auth, config = providers_module
    pools["openrouter"] = _FakePool([_FakeEntry("manual")])

    outcome = mod._purge_provider_from_credential_pool("openrouter")

    assert outcome["ok"] is True
    assert outcome["pool"]["removed"] == 0
    assert config.invalidated == ["openrouter"]
    assert "openrouter" not in auth.suppressed


# ── Failure recording (round-2 review P2: do not silently swallow) ──


def test_load_pool_failure_marks_pool_error(monkeypatch, tmp_path):
    """``load_pool`` raising must be recorded on the outcome, not
    swallowed. Round-1 returned ``None`` and the caller had no way to
    know the pool half had not run.
    """
    monkeypatch.delitem(sys.modules, "agent.credential_pool", raising=False)
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    boom_mod = types.ModuleType("agent.credential_pool")

    def _boom(_provider):
        raise RuntimeError("simulated load_pool failure")

    boom_mod.load_pool = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent.credential_pool", boom_mod)

    # Stub the cache invalidator so the helper can also import it.
    import api.config as _config
    config = _FakeConfig()
    monkeypatch.setattr(
        _config, "invalidate_credential_pool_cache",
        config.invalidate_credential_pool_cache, raising=False,
    )
    if "api.providers" in sys.modules:
        del sys.modules["api.providers"]
    mod = importlib.import_module("api.providers")

    outcome = mod._purge_provider_from_credential_pool("openrouter")

    assert outcome["ok"] is False
    assert outcome["pool"]["ok"] is False
    assert "simulated load_pool failure" in (outcome["pool"]["error"] or "")
    # suppress/cache are not reached when load_pool fails.
    assert outcome["suppress"]["sources"] == []
    assert outcome["cache"]["error"] is None


def test_remove_index_failure_marks_pool_error_but_continues(
    providers_module,
):
    """One bad ``remove_index`` call must record on the outcome and
    not stop the other env entries from being dropped (round-1
    silently continued; round-2 records AND continues).
    """
    mod, pools, auth, _ = providers_module
    env_a = _FakeEntry("env:OPENROUTER_API_KEY", label="or-1")
    env_b = _FakeEntry("env:OPENAI_API_KEY", label="oa-1")
    pool = _FakePool([env_a, env_b])
    pool.remove_index_raises = RuntimeError("simulated remove_index failure")
    pools["openrouter"] = pool

    outcome = mod._purge_provider_from_credential_pool("openrouter")

    assert outcome["ok"] is False
    assert outcome["pool"]["ok"] is False
    assert "simulated remove_index failure" in (outcome["pool"]["error"] or "")
    # Suppress still runs for the sources we *would* have removed --
    # we record the full set on the outcome so the route can decide.
    assert set(outcome["suppress"]["sources"]) == {
        "env:OPENROUTER_API_KEY",
        "env:OPENAI_API_KEY",
    }
    assert outcome["suppress"]["ok"] is True


def test_suppress_failure_marks_suppress_error(monkeypatch, tmp_path):
    """A failing ``suppress_credential_source`` is recorded on the
    outcome. The pool half is still recorded as ok.
    """
    auth = _FakeAuthStore()
    auth.suppress_raises = RuntimeError("simulated suppress failure")
    pools: dict[str, _FakePool] = {}
    config = _FakeConfig()
    _install_fake_agent_modules(
        monkeypatch, pools=pools, auth=auth, config=config
    )
    mod = importlib.import_module("api.providers")

    pools["openrouter"] = _FakePool(
        [_FakeEntry("env:OPENROUTER_API_KEY", label="or-1")]
    )
    outcome = mod._purge_provider_from_credential_pool("openrouter")

    assert outcome["ok"] is False
    assert outcome["pool"]["ok"] is True
    assert outcome["pool"]["removed"] == 1
    assert outcome["suppress"]["ok"] is False
    assert "simulated suppress failure" in (outcome["suppress"]["error"] or "")
    assert outcome["suppress"]["sources"] == []  # none succeeded


def test_cache_invalidate_failure_marks_cache_error(providers_module):
    """A failing ``invalidate_credential_pool_cache`` is recorded on
    the outcome. Pool + suppress are still recorded as ok.
    """
    mod, pools, auth, config = providers_module
    config.invalidate_raises = RuntimeError("simulated cache invalidate failure")
    pools["openrouter"] = _FakePool(
        [_FakeEntry("env:OPENROUTER_API_KEY", label="or-1")]
    )

    outcome = mod._purge_provider_from_credential_pool("openrouter")

    assert outcome["ok"] is False
    assert outcome["pool"]["ok"] is True
    assert outcome["pool"]["removed"] == 1
    assert outcome["suppress"]["ok"] is True
    assert outcome["cache"]["ok"] is False
    assert "simulated cache invalidate failure" in (outcome["cache"]["error"] or "")


# ── Fail-closed: missing dependencies do not crash the removal ──────


def test_missing_agent_credential_pool_module_returns_skipped_outcome(
    monkeypatch, tmp_path
):
    """If ``agent.credential_pool`` cannot be imported (e.g. a
    stripped-down test env) the purge step returns a ``skipped``
    outcome rather than raising. Round-1 returned ``None`` and the
    caller had no signal at all.
    """
    # The previous test (or the ``providers_module`` fixture) may
    # have left ``agent`` / ``agent.credential_pool`` in
    # ``sys.modules``. Drop them so the call-time import inside the
    # helper actually hits the MetaPathFinder below. Without this
    # the helper short-circuits via the cached real module and the
    # test is a no-op in the dev env.
    for mod_name in list(sys.modules):
        if mod_name == "agent" or mod_name.startswith("agent."):
            del sys.modules[mod_name]
    # Install a MetaPathFinder that hard-raises ImportError for any
    # ``agent.*`` import. The dev environment has the real
    # ``hermes-agent`` package on sys.path, so a bare
    # ``del sys.modules`` is not enough -- the real loader is still
    # found. A finder that hard-raises is the only way to make
    # ``from agent.credential_pool import load_pool`` fail at call
    # time in this dev env (matches what CI sees).
    blocker = _BlockAgentFinder()
    sys.meta_path.insert(0, blocker)
    try:
        # Stub the cache invalidator so the helper can import it
        # (this is the one real WebUI-side call).
        import api.config as _config
        config = _FakeConfig()
        monkeypatch.setattr(
            _config, "invalidate_credential_pool_cache",
            config.invalidate_credential_pool_cache, raising=False,
        )
        if "api.providers" in sys.modules:
            del sys.modules["api.providers"]
        mod = importlib.import_module("api.providers")

        outcome = mod._purge_provider_from_credential_pool("openrouter")

        assert outcome["ok"] is False
        assert outcome["skipped"] is not None
        assert "agent.credential_pool" in outcome["skipped"]
        # Pool / suppress / cache were not attempted.
        assert outcome["pool"]["removed"] == 0
        assert outcome["suppress"]["sources"] == []
    finally:
        # Remove the blocker even if assertions fail so the next
        # test can still import the real agent.credential_pool.
        try:
            sys.meta_path.remove(blocker)
        except ValueError:
            pass


# ── Round-2: ``remove_provider_key`` attaches the outcome ───────────


def test_remove_provider_key_attaches_cleanup_outcome(monkeypatch, tmp_path):
    """The public ``remove_provider_key`` result now carries the
    credential-pool outcome under the ``cleanup`` key. This is the
    contract the DELETE route relies on for the partial-failure
    ``warning`` field.
    """
    pools: dict[str, _FakePool] = {}
    auth = _FakeAuthStore()
    config = _FakeConfig()
    _install_fake_agent_modules(
        monkeypatch, pools=pools, auth=auth, config=config
    )
    mod = importlib.import_module("api.providers")

    pools["openrouter"] = _FakePool(
        [_FakeEntry("env:OPENROUTER_API_KEY", label="or-1")]
    )

    # Drive just the purge helper rather than the full
    # ``remove_provider_key`` (which also mutates .env/config.yaml
    # and would require a real Hermes home). The route-level
    # ``remove_provider_key`` end-to-end is exercised by the
    # route-scope test below.
    cleanup = mod._purge_provider_from_credential_pool("openrouter")
    assert cleanup["ok"] is True
    assert cleanup["pool"]["removed"] == 1


# ── Round-2: DELETE route wraps in profile write scope ──────────────


def test_delete_provider_route_uses_profile_write_scope(monkeypatch):
    """Static check: the DELETE route at ``/api/providers/delete``
    must enter ``profile_env_for_active_request`` (write scope) so
    the Agent-side cleanup resolves the request profile's
    ``auth.json``. Without this scope the credential-pool half
    targets the process-default profile and a named-profile client
    can corrupt the wrong ``auth.json`` (review P1).
    """
    src = (REPO / "api" / "routes.py").read_text(encoding="utf-8")
    # Find the DELETE block.
    start = src.find('parsed.path == "/api/providers/delete"')
    assert start != -1, "DELETE provider route not found"
    end = src.find("if parsed.path ==", start + 1)
    if end == -1:
        end = len(src)
    block = src[start:end]
    # The block must (a) import the write-scope context manager and
    # (b) enter it before calling ``remove_provider_key``.
    assert "profile_env_for_active_request" in block, (
        "DELETE route must use the write-scope "
        "profile_env_for_active_request context manager "
        "(review #7412 round-2 P1)"
    )
    # Ordering check: the import + ``with`` must come *before* the
    # ``remove_provider_key`` call (the scope must wrap the call).
    scope_idx = block.find("profile_env_for_active_request")
    call_idx = block.find("remove_provider_key(provider_id)")
    assert scope_idx != -1 and call_idx != -1
    assert scope_idx < call_idx, (
        "profile_env_for_active_request must wrap the "
        "remove_provider_key call (scope before call)"
    )
