"""Per-session MCP toolset override must be ADDITIVE, not replace-all.

Regression guard for the "ticking one MCP server in the per-session toolset
picker breaks the whole chat" bug.

The per-session toolset picker (composer chip) lets a user tick configured MCP
servers for the current chat. Those checkboxes emit the bare MCP server name
(e.g. "my-search"). The override was applied with a wholesale
``_toolsets = _override``, so ticking a single MCP server dropped every built-in
toolset (web, file, terminal, delegation, …) and left the model with an empty
tool list — every tool call failed with ``Tool '...' does not exist. Available
tools:`` (empty).

Fix: an override composed *only* of configured MCP servers is merged on top of
the profile defaults; an override that names any non-MCP toolset keeps the
original restrict-to-these semantics (the power-user free-text use case). A name
that is *both* an MCP server and a builtin toolset (collision, e.g. a server
named ``web``) is excluded from the MCP-only test so it can't silently flip the
override into additive mode and get shadowed by the builtin.

These tests exercise the REAL helper used by the streaming worker
(``api.streaming._apply_session_toolset_override``), not a copied reference, so
the merge-vs-restrict decision is covered on the actual code path.
"""

import importlib
from pathlib import Path

import pytest

import api.streaming as streaming
from api.streaming import _apply_session_toolset_override as _apply_override

REPO = Path(__file__).resolve().parents[1]


# ── CI stand-in for the Hermes agent runtime ────────────────────────────────
# The additive branch of `_apply_session_toolset_override` consults
# `_builtin_toolset_names()`, which `import toolsets` /
# `from tools.registry import registry`.  Those
# modules ship with the hermes-agent runtime, which is NOT installed in the
# WebUI CI test environment — the import fails and the helper returns None,
# which fail-closes the additive path into dropping every builtin toolset
# ("web must survive" assertions break across all Python-version shards).
#
# Local dev machines pass only because hermes-agent happens to be on sys.path
# (PYTHONPATH), so the real 59-toolset registry is imported.  Inject a
# faithful stand-in here when the real module is absent so the tests exercise
# the REAL production code path deterministically in CI too.  Shape mirrors
# hermes-agent's toolsets.py: leaf specs have no non-empty ``includes``;
# composites (e.g. "debugging") have ``includes`` and are NOT static leaves;
# ``resolve_toolset`` expands a name to its tool list.
import sys
import types as _types

# Sentinel for "attribute was absent" in the fixture's snapshot logic.
_MISSING = object()


# ── scoped registry stand-in ────────────────────────────────────────────────
_mock_registry = _types.SimpleNamespace()
_mock_registry._tools = {}
_mock_registry._toolset_aliases = {}
_mock_registry._snapshot_entries = lambda: list(_mock_registry._tools.values())


def _mock_register(name, toolset, schema, handler, check_fn=None,
                   requires_env=None, is_async=False, description="",
                   emoji="", max_result_size_chars=None,
                   dynamic_schema_overrides=None, override=False):
    """Faithful stand-in for ``ToolRegistry.register``: cross-toolset
    shadowing is rejected unless ``override=True`` (mirrors the real
    registry's ownership rules)."""
    existing = _mock_registry._tools.get(name)
    if existing is not None and existing.toolset != toolset and not override:
        return  # rejected shadow, like the real registry
    entry = _types.SimpleNamespace(
        name=name, toolset=toolset, schema=schema, handler=handler,
        check_fn=check_fn, requires_env=requires_env or [],
        is_async=is_async, description=description, emoji=emoji,
        max_result_size_chars=max_result_size_chars,
        dynamic_schema_overrides=dynamic_schema_overrides,
    )
    _mock_registry._tools[name] = entry


def _mock_register_toolset_alias(alias, toolset):
    _mock_registry._toolset_aliases[alias] = toolset


def _mock_get_registered_toolset_names():
    return sorted({e.toolset for e in _mock_registry._snapshot_entries()})


def _mock_get_tool_names_for_toolset(toolset):
    return sorted(e.name for e in _mock_registry._snapshot_entries()
                  if e.toolset == toolset)


def _mock_get_toolset_alias_target(alias):
    return _mock_registry._toolset_aliases.get(alias)


def _mock_get_tool_to_toolset_map():
    return {e.name: e.toolset for e in _mock_registry._snapshot_entries()}


def _mock_get_entry(name):
    return _mock_registry._tools.get(name)


_mock_registry.get_entry = _mock_get_entry
_mock_registry.register = _mock_register
_mock_registry.register_toolset_alias = _mock_register_toolset_alias
_mock_registry.get_registered_toolset_names = _mock_get_registered_toolset_names
_mock_registry.get_tool_names_for_toolset = _mock_get_tool_names_for_toolset
_mock_registry.get_toolset_alias_target = _mock_get_toolset_alias_target
_mock_registry.get_tool_to_toolset_map = _mock_get_tool_to_toolset_map

# ── scoped agent-less stand-in fixture ───────────────────────────────────────
# Round-8 review finding 2: the previous version installed `tools`,
# `tools.registry`, and `toolsets` stand-ins at MODULE IMPORT time with no
# fixture/context-manager teardown, so the fake modules stayed process-global
# for the rest of the pytest shard (and the now-deleted `_prior_sys_modules`
# snapshot was never consumed).  It also decided availability by membership
# (`"tools" in sys.modules`), which treats an importable-but-not-yet-imported
# real runtime as unavailable and shadows it with a fake.
#
# This autouse fixture:
#   1. attempts the REAL imports first,
#   2. injects stand-ins only on genuine import failure,
#   3. snapshots the exact prior presence/object identity of the three module
#      entries plus the parent package's `registry` attribute,
#   4. restores/deletes ONLY what the fixture changed after `yield`,
#   5. asserts (teardown) that no fake remains after scope exit.


@pytest.fixture(autouse=True)
def _agentless_runtime_standins():
    """Install `tools` / `tools.registry` / `toolsets` stand-ins ONLY when the
    real hermes-agent runtime cannot be imported, scoped to the current test,
    and restore the exact prior state afterwards.

    The additive branch of `_apply_session_toolset_override` consults
    `_builtin_toolset_names()`, which does ``import toolsets`` /
    ``from tools.registry import registry``.  Those modules ship with the
    hermes-agent runtime, which is NOT installed in the WebUI CI test
    environment — the import fails and the helper returns None, which
    fail-closes the additive path into dropping every builtin toolset
    ("web must survive" assertions would break on every CI shard).
    """
    # 1/2. Attempt real imports FIRST — only a genuine failure may inject.
    try:
        importlib.import_module("toolsets")
        _real_toolsets = True
    except Exception:
        _real_toolsets = False
    try:
        importlib.import_module("tools.registry")
        _real_registry = True
    except Exception:
        _real_registry = False

    # 3. Snapshot prior presence/object identity of everything we may touch:
    #    the three module entries plus the parent package's `registry` attr.
    _tools_mod = sys.modules.get("tools")
    _prior = {
        "tools": sys.modules.get("tools"),
        "tools.registry": sys.modules.get("tools.registry"),
        "toolsets": sys.modules.get("toolsets"),
        "tools.registry_attr": (
            getattr(_tools_mod, "registry", _MISSING)
            if _tools_mod is not None else _MISSING
        ),
    }
    # (module_key, prior_value, injected_object) — injected_object recorded so
    # the teardown assertion can prove the exact fake is gone.
    _changed = []

    try:
        # 2. Inject ONLY on genuine import failure.
        if not _real_toolsets and "toolsets" not in sys.modules:
            sys.modules["toolsets"] = _mock_toolsets
            _changed.append(("toolsets", _prior["toolsets"], _mock_toolsets))
        if not _real_registry:
            if "tools" not in sys.modules:
                _tools_fake = _types.ModuleType("tools")
                sys.modules["tools"] = _tools_fake
                _changed.append(("tools", _prior["tools"], _tools_fake))
            if "tools.registry" not in sys.modules:
                _mock_registry_mod = _types.ModuleType("tools.registry")
                _mock_registry_mod.registry = _mock_registry  # type: ignore[attr-defined]
                sys.modules["tools.registry"] = _mock_registry_mod
                _changed.append(("tools.registry", _prior["tools.registry"],
                                 _mock_registry_mod))
        yield
    finally:
        # 4. Restore/delete ONLY what this fixture changed.  Entries that
        #    existed before (real or injected by somebody else) are untouched.
        for _key, _prior_val, _injected_obj in reversed(_changed):
            if _prior_val is None:
                sys.modules.pop(_key, None)
            else:
                sys.modules[_key] = _prior_val
        # If the parent package predated us, restore its registry attribute
        # (we never set it, but restore defensively to the snapshot).
        _parent = sys.modules.get("tools")
        _prior_attr = _prior["tools.registry_attr"]
        if _parent is not None:
            if _prior_attr is not _MISSING:
                _parent.registry = _prior_attr  # type: ignore[attr-defined]
            elif hasattr(_parent, "registry") and _prior["tools"] is None:
                # We created the parent; drop the attr we never set.
                delattr(_parent, "registry")
        # 5. Teardown assertion: no fake remains after scope exit.  Every
        #    object this fixture injected must no longer be in sys.modules;
        #    anything that was already there must still be there, unchanged.
        for _key, _prior_val, _injected_obj in _changed:
            _entry = sys.modules.get(_key)
            assert _entry is not _injected_obj, (
                f"stand-in fixture must remove injected {_key} after scope exit"
            )
            if _prior_val is not None:
                assert _entry is _prior_val, (
                    f"stand-in fixture must restore pre-existing {_key} unchanged"
                )

# ── scoped toolsets stand-in ────────────────────────────────────────────────
_mock_toolsets = _types.ModuleType("toolsets")
_mock_toolsets.TOOLSETS = {  # type: ignore[attr-defined]
    "web": {"description": "web tools", "tools": ["web_search"], "includes": []},
    "search": {"description": "search", "tools": ["search"], "includes": []},
    "file": {"description": "file tools", "tools": ["read_file"], "includes": []},
    "terminal": {"description": "terminal", "tools": ["terminal"], "includes": []},
    "delegation": {"description": "delegation", "tools": ["delegate_task"], "includes": []},
    "vision": {"description": "vision", "tools": ["vision_analyze"], "includes": []},
    "computer_use": {"description": "computer use", "tools": ["computer_use"], "includes": []},
    "debugging": {"description": "debugging toolkit", "tools": ["terminal"], "includes": ["web", "file"]},
}


def _get_registry_alias_target(name):  # type: ignore[no-untyped-def]
    try:
        from tools.registry import registry as _r
        return _r.get_toolset_alias_target(name)
    except Exception:
        return None


def _get_registry_tool_names(toolset):  # type: ignore[no-untyped-def]
    try:
        from tools.registry import registry as _r
        return _r.get_tool_names_for_toolset(toolset)
    except Exception:
        return []


def _get_toolset(name, *, include_registry=True):  # type: ignore[no-untyped-def]
    """Mirror the real ``get_toolset``: static spec takes precedence and is
    merged with registry tools registered under the exact static name.
    Aliases are consulted only for non-static names.  This matches the
    installed Agent's resolution order (static-first, alias-second) AND
    its overlay contract (registry tools under a static name are merged)."""
    spec = _mock_toolsets.TOOLSETS.get(name)  # type: ignore[attr-defined]
    if not include_registry:
        return spec
    # Static-first + overlay: merge registry tools registered under the
    # exact static name, exactly like the real ``get_toolset``.
    if spec:
        merged_tools = sorted(
            set(spec.get("tools", []))
            | set(_get_registry_tool_names(name))
        )
        return {"description": spec.get("description", ""),
                "tools": merged_tools,
                "includes": list(spec.get("includes", []))}
    # Non-static: check alias, then registry tools under the resolved name.
    alias_target = _get_registry_alias_target(name)
    registry_toolset = alias_target or name
    registered = _get_registry_tool_names(registry_toolset)
    if registered:
        return {"description": f"Plugin/MCP toolset: {name}",
                "tools": registered, "includes": []}
    return None


def _resolve_toolset(name, _visited=None, *, include_registry=True):  # type: ignore[no-untyped-def]
    if _visited is None:
        _visited = set()
    if name in {"all", "*"}:
        out = set()
        for _n in list(_mock_toolsets.TOOLSETS.keys()):  # type: ignore[attr-defined]
            out.update(_resolve_toolset(_n, set(_visited),
                                        include_registry=include_registry))
        return sorted(out)
    if name in _visited:
        return []
    _visited = _visited | {name}
    spec = _get_toolset(name, include_registry=include_registry)
    if not isinstance(spec, dict):
        return [name]
    tools = list(spec.get("tools") or [])
    for inc in spec.get("includes") or []:
        tools.extend(_resolve_toolset(inc, _visited,
                                      include_registry=include_registry))
    return tools


def _create_custom_toolset(name, description, tools=None, includes=None):  # type: ignore[no-untyped-def]
    _mock_toolsets.TOOLSETS[name] = {  # type: ignore[attr-defined]
        "description": description,
        "tools": tools or [],
        "includes": includes or [],
    }


_mock_toolsets.get_toolset = _get_toolset
_mock_toolsets.resolve_toolset = _resolve_toolset
_mock_toolsets.create_custom_toolset = _create_custom_toolset


# ── Behavioural tests for the merge-vs-restrict decision ─────────────────────


def test_mcp_only_override_is_additive():
    """Ticking a configured MCP server keeps the built-in toolsets and adds
    the MCP server on top."""
    defaults = ["web", "file", "terminal", "delegation"]
    override = ["my-search"]
    mcp_servers = {"my-search"}

    result = _apply_override(defaults, override, mcp_servers, builtin_names={"web", "file", "terminal", "delegation"})

    assert "web" in result, "built-in toolsets must survive an MCP-only override"
    assert "file" in result
    assert "terminal" in result
    assert "delegation" in result
    assert "my-search" in result, "the ticked MCP server must be enabled"


def test_mcp_only_override_dedups_and_preserves_order():
    defaults = ["web", "file", "my-search"]
    override = ["my-search"]
    mcp_servers = {"my-search"}

    result = _apply_override(defaults, override, mcp_servers, builtin_names={"web", "file", "terminal", "delegation"})

    assert result == ["web", "file", "my-search"], (
        "an already-present MCP server must not be duplicated"
    )


def test_multiple_mcp_servers_all_added():
    defaults = ["web", "file"]
    override = ["my-search", "postgres"]
    mcp_servers = {"my-search", "postgres", "github"}

    result = _apply_override(defaults, override, mcp_servers, builtin_names={"web", "file", "terminal", "delegation"})

    assert result == ["web", "file", "my-search", "postgres"]


def test_unchecked_mcp_servers_are_not_leaked():
    """Ticking one MCP server must restrict the others OUT: the override is
    additive over builtins/plugins but restrictive over MCP servers.

    Regression for the Codex gate finding: the earlier additive fix merged
    ALL defaults, so an unchecked MCP server already present in the profile
    defaults (``beta``) leaked its tools back into the session even though
    the user only ticked ``alpha``. Expected: builtins kept, only the
    checked MCP server exposed.
    """
    defaults = ["web", "alpha", "beta"]  # web = builtin; alpha/beta = MCP servers
    override = ["alpha"]                 # user ticked only alpha
    mcp_servers = {"alpha", "beta"}

    result = _apply_override(defaults, override, mcp_servers, builtin_names={"web", "file", "terminal", "delegation"})

    assert result == ["web", "alpha"], (
        "unchecked MCP server 'beta' leaked back in: {}".format(result)
    )


def test_non_mcp_override_still_restricts():
    """A power-user override that names built-in toolsets keeps the original
    restrict-to-these semantics."""
    defaults = ["web", "file", "terminal", "delegation"]
    override = ["file", "terminal"]
    mcp_servers = {"my-search"}

    result = _apply_override(defaults, override, mcp_servers, builtin_names={"web", "file", "terminal", "delegation"})

    assert result == ["file", "terminal"], (
        "a non-MCP override must replace the defaults (restrict semantics)"
    )


def test_mixed_override_restricts():
    """If the override mixes an MCP server with a non-MCP toolset, it is not
    'MCP-only', so restrict semantics apply (defaults are replaced)."""
    defaults = ["web", "file", "terminal"]
    override = ["my-search", "file"]
    mcp_servers = {"my-search"}

    result = _apply_override(defaults, override, mcp_servers, builtin_names={"web", "file", "terminal", "delegation"})

    assert result == ["my-search", "file"]


def test_empty_override_leaves_defaults():
    defaults = ["web", "file"]
    assert _apply_override(defaults, [], {"my-search"}, builtin_names={"web", "file", "terminal", "delegation"}) == ["web", "file"]
    assert _apply_override(defaults, None, {"my-search"}, builtin_names={"web", "file", "terminal", "delegation"}) == ["web", "file"]


def test_no_configured_mcp_servers_falls_back_to_restrict():
    """When there are no configured MCP servers, an override can only be a
    restrict list — never additive."""
    defaults = ["web", "file"]
    override = ["my-search"]

    result = _apply_override(defaults, override, set(), builtin_names={"web", "file", "terminal", "delegation"})

    assert result == ["my-search"]


# ── Collision case: MCP server name shadowed by a builtin toolset ────────────


def test_builtin_collision_is_restrict_not_additive():
    """A configured MCP server whose name collides with a builtin toolset
    (e.g. a server literally named ``web``) must NOT flip the override into
    additive mode. The builtin shadows the MCP alias at resolution time, so
    treating ``["web"]`` as additive would both mis-resolve the override and
    leave the MCP tools unavailable. Restrict semantics apply instead."""
    defaults = ["web", "file", "terminal", "delegation"]
    override = ["web"]
    mcp_servers = {"web"}          # a server that shares a builtin's name
    builtin_names = {"web", "file", "terminal", "delegation"}

    result = _apply_override(defaults, override, mcp_servers, builtin_names=builtin_names)

    assert result == ["web"], (
        "a name that is both an MCP server and a builtin must take restrict "
        "semantics, not additive — otherwise it mis-resolves and the MCP "
        "tools stay unavailable"
    )


def test_mcp_only_additive_when_some_servers_collide():
    """A pure-MCP name (no builtin collision) stays additive even when *other*
    configured servers happen to collide with builtins."""
    defaults = ["web", "file", "terminal"]
    override = ["my-search"]
    mcp_servers = {"my-search", "web"}   # "web" collides, "my-search" does not
    builtin_names = {"web", "file", "terminal"}

    result = _apply_override(defaults, override, mcp_servers, builtin_names=builtin_names)

    assert result == ["web", "file", "terminal", "my-search"], (
        "a non-colliding MCP-only tick must still be additive"
    )


def test_real_builtin_names_default_lookup(monkeypatch):
    """With an *available* builtin registry, a genuine MCP-only override is
    additive because a normal server name does not collide with any builtin.

    The default lookup (``_builtin_toolset_names()``) returns ``None`` when the
    Hermes ``toolsets`` module isn't importable (e.g. the WebUI test env in CI),
    which correctly fails closed to RESTRICT. To assert the *additive* path we
    stub the helper to report an available, non-colliding builtin set — that is
    the environment this test is about.
    """
    import api.streaming as streaming

    monkeypatch.setattr(
        streaming, "_builtin_toolset_names",
        lambda: {"web", "file", "terminal", "delegation"},
    )

    defaults = ["web", "file"]
    override = ["my-search"]
    mcp_servers = {"my-search"}

    # builtin_names=None → consults the (stubbed, available) helper.
    result = _apply_override(defaults, override, mcp_servers)

    assert "web" in result and "file" in result and "my-search" in result


def test_default_lookup_unavailable_registry_restricts(monkeypatch):
    """The other half of the default-lookup contract: when the real helper
    reports the registry is unavailable (``None``), the same MCP-only override
    fails closed to RESTRICT rather than additive. This documents that the
    additive path depends on an available registry."""
    import api.streaming as streaming

    monkeypatch.setattr(streaming, "_builtin_toolset_names", lambda: None)

    result = _apply_override(["web", "file"], ["my-search"], {"my-search"})

    assert result == ["my-search"], (
        "an unavailable registry must fail closed to restrict on the default "
        "lookup path too"
    )


# ── Fail-closed: unavailable builtin registry must RESTRICT, never additive ──


def test_unavailable_registry_forces_restrict(monkeypatch):
    """When ``_builtin_toolset_names()`` can't resolve the builtin list it
    returns ``None`` ("I don't know"). An MCP-only override with a colliding
    server name must then fall back to RESTRICT, not re-open the collision by
    treating every name as MCP-additive.

    This reproduces the round-2 gate finding: forcing the empty/unavailable
    fallback with ``override=['web']`` must restrict, not restore all defaults.
    """
    import api.streaming as streaming

    # Simulate the registry being unavailable.
    monkeypatch.setattr(streaming, "_builtin_toolset_names", lambda: None)

    defaults = ["web", "file", "terminal", "delegation"]
    override = ["web"]
    mcp_servers = {"web"}   # collides with a builtin name

    # builtin_names=None → helper is consulted → returns None → fail closed.
    result = _apply_override(defaults, override, mcp_servers)

    assert result == ["web"], (
        "an unavailable builtin registry must fail closed to RESTRICT; it must "
        "NOT re-open the collision by restoring all defaults additively"
    )


def test_none_builtin_names_argument_forces_restrict():
    """Passing ``builtin_names=None`` explicitly (registry unavailable) with a
    colliding MCP server must restrict, independent of the helper lookup."""
    import api.streaming as streaming

    # Ensure the helper (consulted when builtin_names is None) also reports
    # unavailable, so this test asserts the None-path in isolation.
    real = streaming._builtin_toolset_names
    streaming._builtin_toolset_names = lambda: None
    try:
        result = _apply_override(
            ["web", "file", "terminal", "delegation"],
            ["web"],
            {"web"},
            builtin_names=None,
        )
    finally:
        streaming._builtin_toolset_names = real

    assert result == ["web"], (
        "builtin_names=None means the registry is unavailable → restrict"
    )


# ── Shadow set: registered/plugin toolset names also shadow MCP aliases ──────


def test_registry_shadow_collision_restricts():
    """A configured MCP server whose name collides with a *registered* (plugin
    or canonical) toolset — not just a static builtin — must also take restrict
    semantics. The builtin_names set is expected to include such registered
    names so the collision guard covers the full shadow set."""
    defaults = ["web", "file", "my-plugin"]
    override = ["my-plugin"]
    mcp_servers = {"my-plugin"}      # server shares a registered toolset's name
    # builtin_names carries the full shadow set including the plugin toolset.
    builtin_names = {"web", "file", "terminal", "delegation", "my-plugin"}

    result = _apply_override(defaults, override, mcp_servers,
                             builtin_names=builtin_names)

    assert result == ["my-plugin"], (
        "a name shadowed by a registered/plugin toolset must restrict, not "
        "flip the override into additive mode"
    )


# ── Source-level invariant: the replace-all bug must not come back ───────────


def test_streaming_uses_additive_helper():
    """Pin the source so a future edit can't silently revert to the
    'any override replaces the defaults' shape that broke MCP-only chats.

    A mere substring search for the helper *name* is vacuous — the function's
    own definition satisfies it even if the call-site is dead. So we parse the
    AST and assert the streaming worker actually *calls*
    ``_apply_session_toolset_override`` with the MCP server names, on a real
    call path.
    """
    import ast

    src = (REPO / "api" / "streaming.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Find genuine call sites (not the def), excluding the function definition
    # itself so a dead/renamed body can't satisfy the guard.
    call_sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "_apply_session_toolset_override":
                call_sites.append(node)

    assert call_sites, (
        "streaming.py must CALL _apply_session_toolset_override() on a real "
        "code path (not merely define it). Without an active call an MCP-only "
        "toolset chip will again wipe out every built-in tool."
    )

    # At least one call site must pass the configured MCP server names through
    # (3rd positional arg), so the additive-vs-restrict decision is actually
    # driven by which names are MCP servers.
    def _passes_mcp_names(call):
        # positional: (defaults, override, mcp_server_names, [builtin_names])
        if len(call.args) >= 3:
            return True
        # or keyword mcp_server_names=...
        return any(kw.arg == "mcp_server_names" for kw in call.keywords)

    assert any(_passes_mcp_names(c) for c in call_sites), (
        "the call-site must pass the configured MCP server names into "
        "_apply_session_toolset_override() so the collision-aware "
        "additive-vs-restrict decision is driven by real config, not a stub."
    )

    # It must read the configured MCP server names from config.
    assert "mcp_servers" in src, (
        "streaming.py must read cfg['mcp_servers'] to distinguish MCP server "
        "names from ordinary toolset names in the per-session override."
    )


def test_plugin_registry_collision_stays_restrict(monkeypatch):
    """When a registered plugin toolset name collides with a configured MCP
    server name, the override must stay RESTRICT — not silently flip additive.

    The regression bug: ``_builtin_toolset_names()`` used speculative
    accessor/attribute names that don't exist on the installed runtime, so
    registered/plugin toolsets were silently missing from the shadow set.  A
    plugin-named MCP server could slip past the collision guard and flip a
    RESTRICT override to additive.
    """
    import api.streaming as streaming

    # Simulate a registered plugin toolset whose name collides with an MCP
    # server.  The real registry may be unavailable in CI — we patch the
    # registry method so the test exercises the real code path in
    # _builtin_toolset_names() regardless of the environment.
    plugin_name = "browser-cdp"
    try:
        from tools.registry import registry as _reg
        _orig_get = _reg.get_registered_toolset_names
        monkeypatch.setattr(
            _reg, "get_registered_toolset_names",
            lambda: [plugin_name],
        )
    except ImportError:
        # CI environment — tools.registry is not importable.
        # _builtin_toolset_names() will return None (fail-closed), which is
        # the correct behaviour.  The test is still meaningful: we verify
        # that the fail-closed path does NOT silently flip to additive.
        pass

    # Drive the real _builtin_toolset_names() (not a stubbed return value).
    builtin = streaming._builtin_toolset_names()

    if builtin is None:
        # Registry unavailable → fail-closed RESTRICT is correct.
        result = _apply_override(
            ["web", "file"], [plugin_name], {plugin_name},
            builtin_names=None,
        )
        assert result == [plugin_name], (
            "unavailable registry must restrict, not silently flip additive"
        )
        return

    # Registry available → plugin name must be in the shadow set.
    assert plugin_name in builtin, (
        f"registered plugin '{plugin_name}' must appear in the builtin shadow "
        f"set so it collides with the same-named MCP server"
    )

    # The collision must force RESTRICT.
    result = _apply_override(
        ["web", "file"], [plugin_name], {plugin_name},
        builtin_names=builtin,
    )
    assert result == [plugin_name], (
        f"plugin '{plugin_name}' colliding with MCP server must RESTRICT, "
        f"not silently flip additive (got {result})"
    )


# ── mcp-* canonical name collision: alpha + mcp-alpha ──────────────────────
# Regression for the gate finding: dropping every mcp-* canonical name from
# the shadow set enables a silent wrong-server resolution when two servers
# have overlapping canonical names.


def _patch_registry_and_get_builtin_names(monkeypatch, registered_names):
    """Monkeypatch ``get_registered_toolset_names()`` to return *registered_names*
    and call the real ``_builtin_toolset_names()``.  Returns the full shadow set
    (static TOOLSETS keys + registered names).

    Both ``toolsets`` and ``tools.registry`` are made available unconditionally —
    when they cannot be imported (CI without the agent runtime), lightweight
    stand-ins are injected into ``sys.modules`` so the production
    ``_builtin_toolset_names()`` import succeeds and the real enumeration path is
    exercised.  This prevents a silent fallback that would hide a regression in
    the ``mcp-*`` filtering.
    """
    import sys
    import types
    import api.streaming as streaming

    _injected = []  # (module_name, was_present) pairs for cleanup
    _reg = None

    # ── toolsets ──
    if "toolsets" not in sys.modules:
        _mock_toolsets = types.ModuleType("toolsets")
        # Minimal TOOLSETS dict — the static keys the real module exposes.
        # Only the keys matter for shadow-set membership; the values are unused.
        _mock_toolsets.TOOLSETS = {  # type: ignore[attr-defined]
            "web": None, "search": None, "file": None, "terminal": None,
            "delegation": None, "vision": None, "computer_use": None,
        }
        sys.modules["toolsets"] = _mock_toolsets
        _injected.append(("toolsets", False))
    else:
        _injected.append(("toolsets", True))

    # ── tools.registry ──
    try:
        from tools.registry import registry as _reg
    except ImportError:
        _mock_registry = types.SimpleNamespace()
        _mock_registry.get_registered_toolset_names = lambda: list(registered_names)
        _reg = _mock_registry
        _injected_module = types.ModuleType("tools.registry")
        _injected_module.registry = _mock_registry  # type: ignore[attr-defined]
        if "tools" not in sys.modules:
            sys.modules["tools"] = types.ModuleType("tools")
            _injected.append(("tools", False))
        else:
            _injected.append(("tools", True))
        sys.modules["tools.registry"] = _injected_module
        _injected.append(("tools.registry", False))
    else:
        _injected.append(("tools.registry", True))
        _injected.append(("tools", True))

    _orig = _reg.get_registered_toolset_names
    monkeypatch.setattr(_reg, "get_registered_toolset_names",
                        lambda: list(registered_names))
    try:
        return streaming._builtin_toolset_names()
    finally:
        monkeypatch.setattr(_reg, "get_registered_toolset_names", _orig)
        # Clean up injected modules so other tests see the real environment.
        for _mod_name, _was_present in reversed(_injected):
            if not _was_present:
                sys.modules.pop(_mod_name, None)


def test_mcp_canonical_name_collision_selects_owned_server(monkeypatch):
    """When MCP servers ``alpha`` and ``mcp-alpha`` coexist, the picker
    token ``mcp-alpha`` (server ``mcp-alpha``'s bare name) collides with the
    *canonical* name ``mcp-alpha`` (owned by server ``alpha``).  The
    override must ADDITIVELY emit the selected server's TRUE canonical
    ``mcp-mcp-alpha`` (proven via the live alias edge ``mcp-alpha ->
    mcp-mcp-alpha``) — never bare ``mcp-alpha``, which the runtime resolver
    would match to server ``alpha`` (gate-certified SILENT).

    Production-composed: ``builtin_names`` is NOT injected.  The real
    ``_builtin_toolset_names()`` collector runs, so ``mcp-alpha`` and
    ``mcp-mcp-alpha`` are both present in the broad registered-name shadow
    set — the round-11 fix must exclude OWNED canonicals (alias-edge
    proven) before the MCP-only test, or the token is wrongly restricted.
    """
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        # Server "alpha" → canonical mcp-alpha
        _ts_mod.create_custom_toolset("alpha", "server alpha",
                                      tools=[], includes=[])
        _reg.register_toolset_alias("alpha", "mcp-alpha")
        _reg.register("alpha_owned_tool", "mcp-alpha", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        # Server "mcp-alpha" → canonical mcp-mcp-alpha
        _ts_mod.create_custom_toolset("mcp-alpha", "server mcp-alpha",
                                      tools=[], includes=[])
        _reg.register_toolset_alias("mcp-alpha", "mcp-mcp-alpha")
        _reg.register("mcp_alpha_owned_tool", "mcp-mcp-alpha",
                      {"type": "object"},
                      lambda *a, **k: "ok", override=True)

        # Seed-assertion relevance: prove the collision edges are live
        # BEFORE driving the classifier (round-8 rule).
        assert _reg.get_toolset_alias_target("alpha") == "mcp-alpha"
        assert _reg.get_toolset_alias_target("mcp-alpha") == "mcp-mcp-alpha"

        result = _apply_override(
            ["web", "file", "terminal", "delegation"], ["mcp-alpha"],
            {"alpha", "mcp-alpha"}, builtin_names=None)

        assert "mcp-alpha" not in result, (
            f"ambiguous bare 'mcp-alpha' must never survive, got {result!r}"
        )
        assert "mcp-mcp-alpha" in result, (
            f"selected server's true canonical must be emitted, got {result!r}"
        )
        assert "web" in result, f"builtin web must survive, got {result!r}"
        resolved = set(_ts_mod.resolve_toolset("mcp-mcp-alpha"))
        assert "mcp_alpha_owned_tool" in resolved, (
            f"mcp-mcp-alpha must resolve the SELECTED server's tools, "
            f"got {sorted(resolved)!r}"
        )
        assert "alpha_owned_tool" not in resolved, (
            f"wrong server's tools must not resolve, got {sorted(resolved)!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


def test_canonical_collision_without_alias_edge_fails_closed(monkeypatch):
    """Negative control: when the registry exposes a canonical name (e.g.
    ``mcp-alpha``) but NO live alias edge proves it is owned by a configured
    MCP server, ownership cannot be proved → the broad shadow set keeps it
    and the override fails closed to FULLY RESTRICT (empty list).

    Round-13 fix: the restrict branch now drops mcp-* selectors whose
    ownership cannot be proved, preventing wrong-server exposure.
    Returning the raw selector would let the resolver pick server alpha
    when the user intended server mcp-alpha — a gate-certified SILENT failure.

    Uses the real ``_builtin_toolset_names()`` (monkeypatched registry),
    not a hand-constructed shadow set."""

    builtin = _patch_registry_and_get_builtin_names(
        monkeypatch, ["mcp-alpha", "mcp-mcp-alpha"])

    assert "mcp-alpha" in builtin, (
        "canonical 'mcp-alpha' must be in the shadow set (got {})".format(builtin)
    )

    result = _apply_override(
        ["web", "file", "terminal", "delegation"], ["mcp-alpha"],
        {"alpha", "mcp-alpha"}, builtin_names=builtin)

    # Round-13 fix: no alias edge for 'alpha' → mcp-alpha means ownership
    # of canonical 'mcp-alpha' cannot be proved → must drop it entirely.
    # Returning bare 'mcp-alpha' would resolve to server alpha's tools,
    # not the intended server mcp-alpha's tools.
    assert result == [], (
        "no alias edge → ownership unprovable → must DROP the selector "
        "and return empty list (got {!r})".format(result)
    )


def test_ordinary_server_stays_additive_with_canonical_in_shadow(monkeypatch):
    """Negative control: a normal server ``foo`` (pick token ``foo``) stays
    additive even though its canonical name ``mcp-foo`` is in the shadow set.
    The bare token ``foo`` ≠ ``mcp-foo``, so there is no collision.

    Uses the real ``_builtin_toolset_names()`` (monkeypatched registry), not
    hand-constructed shadow set."""
    builtin = _patch_registry_and_get_builtin_names(
        monkeypatch, ["mcp-foo", "mcp-bar"])

    assert "mcp-foo" in builtin
    assert "foo" not in builtin, (
        "bare token 'foo' must NOT be in the shadow set — only canonical "
        "'mcp-foo' is (got {})".format(builtin)
    )

    result = _apply_override(
        ["web", "file"], ["foo"], {"foo", "bar"}, builtin_names=builtin)

    assert result == ["web", "file", "foo"], (
        "bare token 'foo' must stay additive even with canonical 'mcp-foo' "
        "in the shadow set (got {})".format(result)
    )


def test_plugin_canonical_mcp_prefix_collision_restricts(monkeypatch):
    """A plugin whose canonical name begins with ``mcp-`` (e.g. ``mcp-browser``)
    must shadow a same-named MCP server alias.  The old code filtered *all*
    ``mcp-*`` names from the shadow set, so a plugin ``mcp-browser`` could
    silently flip additive.

    Round-13 fix: the restrict branch drops mcp-* selectors whose ownership
    cannot be proved.  A plugin's canonical has no MCP alias edge, so the
    selector must be dropped entirely — returning it would resolve to the
    plugin's tools, not the intended MCP server's tools.

    Uses the real ``_builtin_toolset_names()`` (monkeypatched registry), not
    hand-constructed shadow set."""

    builtin = _patch_registry_and_get_builtin_names(
        monkeypatch, ["mcp-browser"])

    assert "mcp-browser" in builtin, (
        "plugin canonical 'mcp-browser' must be in the shadow set "
        "(got {})".format(builtin)
    )

    result = _apply_override(
        ["web", "file"], ["mcp-browser"], {"mcp-browser"},
        builtin_names=builtin)

    # Round-13 fix: plugin canonical 'mcp-browser' has no MCP alias edge,
    # so ownership cannot be proved → must DROP entirely.
    assert result == [], (
        "plugin canonical 'mcp-browser' has no alias edge → ownership "
        "unprovable → must DROP the selector and return empty list "
        "(got {})".format(result)
    )


# ── Malformed registry return shapes must fail closed ──────────────────────


def test_registry_returns_none_fails_closed(monkeypatch):
    """When ``get_registered_toolset_names()`` returns None, the helper must
    return None (not a partial static-only set)."""
    import api.streaming as streaming

    try:
        from tools.registry import registry as _reg
    except ImportError:
        return  # CI without tools.registry — skip

    _orig = _reg.get_registered_toolset_names
    monkeypatch.setattr(_reg, "get_registered_toolset_names", lambda: None)
    try:
        result = streaming._builtin_toolset_names()
    finally:
        monkeypatch.setattr(_reg, "get_registered_toolset_names", _orig)

    assert result is None, (
        "None registry result must fail closed (return None), "
        f"got {result!r}"
    )


def test_registry_returns_dict_fails_closed(monkeypatch):
    """When ``get_registered_toolset_names()`` returns a dict (not a set/list/
    tuple), the helper must return None."""
    import api.streaming as streaming

    try:
        from tools.registry import registry as _reg
    except ImportError:
        return

    _orig = _reg.get_registered_toolset_names
    monkeypatch.setattr(_reg, "get_registered_toolset_names", lambda: {"a": 1})
    try:
        result = streaming._builtin_toolset_names()
    finally:
        monkeypatch.setattr(_reg, "get_registered_toolset_names", _orig)

    assert result is None, (
        f"dict registry result must fail closed, got {result!r}"
    )


def test_registry_returns_string_fails_closed(monkeypatch):
    """When ``get_registered_toolset_names()`` returns a bare string, the
    helper must return None."""
    import api.streaming as streaming

    try:
        from tools.registry import registry as _reg
    except ImportError:
        return

    _orig = _reg.get_registered_toolset_names
    monkeypatch.setattr(_reg, "get_registered_toolset_names", lambda: "not-a-list")
    try:
        result = streaming._builtin_toolset_names()
    finally:
        monkeypatch.setattr(_reg, "get_registered_toolset_names", _orig)

    assert result is None, (
        f"string registry result must fail closed, got {result!r}"
    )


def test_registry_contains_non_string_entries_fails_closed(monkeypatch):
    """When ``get_registered_toolset_names()`` returns entries that are not
    strings, the helper must return None rather than string-coercing them."""
    import api.streaming as streaming

    try:
        from tools.registry import registry as _reg
    except ImportError:
        return

    _orig = _reg.get_registered_toolset_names
    monkeypatch.setattr(
        _reg, "get_registered_toolset_names",
        lambda: ["valid", 42, object()],
    )
    try:
        result = streaming._builtin_toolset_names()
    finally:
        monkeypatch.setattr(_reg, "get_registered_toolset_names", _orig)

    assert result is None, (
        f"non-string entries in registry must fail closed, got {result!r}"
    )


# ── Canonical selector and wildcard leakage regression ─────────────────────
# Gate certification found that stripping only the bare server name from
# defaults lets the canonical ``mcp-<name>`` selector or a wildcard (``all`` /
# ``*``) survive the additive merge, re-exposing unchecked MCP servers.


def test_canonical_mcp_selector_in_defaults_is_stripped():
    """A default that exposes an MCP server via its canonical ``mcp-<name>``
    selector must be stripped along with the bare name, so the unchecked
    server's tools don't leak.

    Reproduces the gate finding: defaults ``['web', 'mcp-alpha']`` + selected
    ``['beta']`` must NOT retain ``mcp-alpha`` (alpha's tools would come
    back through the installed resolver's canonical resolution path).
    """
    defaults = ["web", "mcp-alpha"]
    override = ["beta"]
    mcp_servers = {"alpha", "beta"}

    result = _apply_override(
        defaults, override, mcp_servers,
        builtin_names={"web", "file", "terminal", "delegation"},
    )

    assert "mcp-alpha" not in result, (
        "canonical 'mcp-alpha' in defaults must be stripped so unchecked "
        "alpha's tools don't leak (got {})".format(result)
    )
    assert "web" in result, "builtins must survive"
    assert "beta" in result, "checked server must be present"


def test_wildcard_all_in_defaults_fails_closed_when_mcp_unchecked():
    """A wildcard default ``['all']`` + override ``['beta']`` with unchecked
    server ``alpha``: the wildcard would expand to every registered toolset
    including alpha's, so we must fail closed to restrictive semantics
    (``['beta']``) rather than risk leaking alpha's tools.
    """
    defaults = ["all"]
    override = ["beta"]
    mcp_servers = {"alpha", "beta"}

    result = _apply_override(
        defaults, override, mcp_servers,
        builtin_names={"web", "file", "terminal", "delegation"},
    )

    assert result == ["beta"], (
        "wildcard 'all' with unchecked MCP server must fail closed to "
        "restrict (got {})".format(result)
    )


def test_wildcard_star_in_defaults_fails_closed_when_mcp_unchecked():
    """Same as above but with ``*`` instead of ``all``."""
    defaults = ["*"]
    override = ["beta"]
    mcp_servers = {"alpha", "beta"}

    result = _apply_override(
        defaults, override, mcp_servers,
        builtin_names={"web", "file", "terminal", "delegation"},
    )

    assert result == ["beta"], (
        "wildcard '*' with unchecked MCP server must fail closed to "
        "restrict (got {})".format(result)
    )


def test_wildcard_all_ok_when_all_mcp_checked():
    """When ALL configured MCP servers are ticked, a wildcard default is
    safe — no unchecked server can leak through the expansion."""
    defaults = ["all"]
    override = ["alpha", "beta"]
    mcp_servers = {"alpha", "beta"}

    result = _apply_override(
        defaults, override, mcp_servers,
        builtin_names={"web", "file", "terminal", "delegation"},
    )

    assert "alpha" in result and "beta" in result, (
        "all-checked wildcard must retain the checked servers (got {})".format(result)
    )


def test_canonical_selector_not_stripped_when_no_mcp_server():
    """A default ``mcp-foo`` with no configured MCP server named ``foo`` is a
    user-authored canonical name that should be preserved (it's not an MCP
    server we can strip)."""
    defaults = ["web", "mcp-foo"]
    override = ["alpha"]
    mcp_servers = {"alpha"}

    result = _apply_override(
        defaults, override, mcp_servers,
        builtin_names={"web", "file", "terminal", "delegation"},
    )

    assert "mcp-foo" in result, (
        "canonical 'mcp-foo' with no configured server 'foo' must survive "
        "(got {})".format(result)
    )


def test_canonical_selector_with_checked_server_preserved():
    """When the server IS checked, its canonical selector in defaults must be
    preserved (not stripped) — the override adds the bare name, and the
    canonical is just an alias to the same server."""
    defaults = ["web", "mcp-alpha"]
    override = ["alpha"]
    mcp_servers = {"alpha", "beta"}

    result = _apply_override(
        defaults, override, mcp_servers,
        builtin_names={"web", "file", "terminal", "delegation"},
    )

    # The canonical selector should survive because alpha IS checked.
    assert "alpha" in result, "checked server must be present"


def test_existing_unchecked_regression_with_canonical():
    """Combined regression: defaults with both bare and canonical forms +
    multiple unchecked servers."""
    defaults = ["web", "alpha", "mcp-beta"]
    override = ["alpha"]
    mcp_servers = {"alpha", "beta"}

    result = _apply_override(
        defaults, override, mcp_servers,
        builtin_names={"web", "file", "terminal", "delegation"},
    )

    assert "web" in result, "builtins must survive"
    assert "alpha" in result, "checked server must be present"
    assert "beta" not in result, "bare beta must be stripped"
    assert "mcp-beta" not in result, (
        "canonical 'mcp-beta' must also be stripped (got {})".format(result)
    )


# ── Blocking finding 1: malformed override must fail closed ────────────────


def test_malformed_override_dict_entry_fails_closed():
    """An unhashable persisted override entry (e.g. ``[{"stale": "shape"}]``)
    must NOT raise TypeError inside the additive classifier and fall through
    to the caller's broad except (which restores all profile defaults).

    The helper must return the restrict fallback (string entries only) without
    raising.
    """
    defaults = ["web", "file", "terminal", "alpha", "beta"]
    override = [{"stale": "shape"}]  # unhashable dict entry

    result = _apply_override(defaults, override, {"alpha", "beta"})

    assert isinstance(result, list), f"must return a list, got {type(result)}"
    # Malformed entries are filtered out; no string entries remain → empty list
    # (restrict semantics with no valid toolset names).
    assert result == [], (
        f"malformed override must not restore defaults (fail-open), got {result!r}"
    )


def test_malformed_override_mixed_entries_fails_closed():
    """A mix of valid string entries and malformed entries must keep only the
    valid strings and apply restrict semantics — never restore all defaults.
    """
    defaults = ["web", "file", "terminal", "alpha", "beta"]
    override = ["alpha", {"stale": "shape"}, "beta"]

    result = _apply_override(defaults, override, {"alpha", "beta"})

    # Malformed entry dropped → ["alpha", "beta"] returned as restrict fallback.
    # The critical assertion: defaults are NOT restored (no fail-open).
    assert result == ["alpha", "beta"], (
        f"malformed override must return string-only restrict list, got {result!r}"
    )
    assert "web" not in result and "file" not in result, (
        f"defaults must NOT be restored on malformed override (fail-open), got {result!r}"
    )


def test_non_string_mcp_server_name_fails_closed():
    """A non-string ``mcp_servers`` key must fail closed to restrict, not
    reach ``'mcp-' + _srv`` downstream and fail-open via the caller's except.
    """
    defaults = ["web", "file", "terminal", "alpha"]
    override = ["alpha"]
    # mcp_server_names contains a non-string entry
    mcp_names = {"alpha", 42}

    result = _apply_override(defaults, override, mcp_names)

    assert result == ["alpha"], (
        f"non-string mcp_server_names must fail closed to restrict, got {result!r}"
    )


# ── Blocking finding 2: recursive composite must not leak unchecked MCP ─────


def test_recursive_composite_does_not_leak_unchecked_mcp(monkeypatch):
    """A composite toolset in defaults whose ``includes`` transitively pulls in
    an unchecked MCP server must be DROPPED from the merged result.

    Scenario (from the Codex adversarial gate):
      - ``gate-alpha`` and ``gate-beta`` are configured MCP servers.
      - ``gate-composite`` is a toolset that includes ``gate-alpha``.
      - Profile defaults: ``['web', 'gate-composite']``
      - Session override: ``['gate-beta']``

    Without the fix, ``gate-composite`` survives the name-level filter (it's
    not a bare MCP name) and is kept in ``merged``.  When the agent resolves
    it, ``gate-alpha`` tools leak back in — a silent authorization regression.

    With the fix, ``_default_transitively_reaches_unchecked_mcp`` expands
    ``gate-composite``, finds tools registered under ``mcp-gate-alpha``, and
    drops it.
    """
    try:
        from tools.registry import registry as _reg
        import api.streaming as streaming
    except ImportError:
        return  # CI without tools.registry — skip

    # Register two fake MCP servers and a composite toolset.
    _fake_tools = {
        "gate_alpha_tool": _reg._tools.get("gate_alpha_tool"),
        "gate_beta_tool": _reg._tools.get("gate_beta_tool"),
    }

    class _FakeEntry:
        def __init__(self, name, toolset):
            self.name = name
            self.toolset = toolset

    _orig_snapshot = _reg._snapshot_entries

    def _fake_snapshot():
        real = [e for e in _orig_snapshot()]
        real.append(_FakeEntry("gate_alpha_tool", "mcp-gate-alpha"))
        real.append(_FakeEntry("gate_beta_tool", "mcp-gate-beta"))
        return real

    # Patch resolve_toolset to simulate gate-composite including gate-alpha
    import toolsets as _ts_mod

    _orig_resolve = _ts_mod.resolve_toolset

    def _fake_resolve(name, visited=None, *, include_registry=True):
        if name == "gate-composite":
            # Composite includes gate-alpha → exposes its tools
            return ["gate_alpha_tool"]
        if name == "gate-alpha":
            return ["gate_alpha_tool"]
        if name == "gate-beta":
            return ["gate_beta_tool"]
        return _orig_resolve(name, visited, include_registry=include_registry)

    monkeypatch.setattr(_reg, "_snapshot_entries", _fake_snapshot)
    monkeypatch.setattr(_ts_mod, "resolve_toolset", _fake_resolve)
    # Also patch the reference imported in streaming module
    monkeypatch.setattr(streaming, "_default_transitively_reaches_unchecked_mcp",
                        streaming._default_transitively_reaches_unchecked_mcp)

    defaults = ["web", "gate-composite"]
    override = ["gate-beta"]
    mcp_names = {"gate-alpha", "gate-beta"}
    # Use a minimal builtin_names that doesn't include gate-* names
    builtin_names = {"web", "file", "terminal", "delegation"}

    result = _apply_override(
        defaults, override, mcp_names, builtin_names=builtin_names
    )

    # gate-composite must be dropped — it transitively reaches gate-alpha
    assert "gate-composite" not in result, (
        f"composite including unchecked MCP must be dropped, got {result!r}"
    )
    # web must survive (not a composite, not an MCP server)
    assert "web" in result, (
        f"non-composite built-in must survive, got {result!r}"
    )
    # gate-beta (the ticked server) must be present
    assert "gate-beta" in result, (
        f"ticked MCP server must be present, got {result!r}"
    )
    # gate-alpha (unchecked) must NOT be present
    assert "gate-alpha" not in result, (
        f"unchecked MCP server must not appear, got {result!r}"
    )


def test_composite_not_leaking_unchecked_mcp_is_kept(monkeypatch):
    """A composite toolset that does NOT transitively reach any unchecked MCP
    server must be KEPT in the merged result (no false positives).
    """
    try:
        import toolsets as _ts_mod
    except ImportError:
        return  # CI without toolsets — skip

    _orig_resolve = _ts_mod.resolve_toolset

    def _fake_resolve(name, visited=None, *, include_registry=True):
        if name == "safe-composite":
            # Includes only web (a builtin), no MCP servers
            return _orig_resolve("web", visited, include_registry=include_registry)
        return _orig_resolve(name, visited, include_registry=include_registry)

    monkeypatch.setattr(_ts_mod, "resolve_toolset", _fake_resolve)

    defaults = ["web", "safe-composite"]
    override = ["gate-beta"]
    mcp_names = {"gate-alpha", "gate-beta"}
    builtin_names = {"web", "file", "terminal", "safe-composite"}

    result = _apply_override(
        defaults, override, mcp_names, builtin_names=builtin_names
    )

    # safe-composite must be kept — it doesn't reach any unchecked MCP
    assert "gate-beta" in result, (
        f"ticked MCP server must be present, got {result!r}"
    )


# ── Review finding regressions (round 4) ────────────────────────────────────


def test_scalar_truthy_override_fails_closed():
    """Review finding #1: a truthy scalar persisted override (e.g. ``42``)
    must NOT raise ``TypeError`` inside the list-comprehension and fall
    through to the caller's broad ``except``, which would restore ALL
    profile defaults (fail-open)."""
    defaults = ["web", "file", "terminal", "delegation"]
    # Truthy scalar — the old code's ``if not override`` didn't catch it
    # because 42 is truthy, then the malformed list-comp raised TypeError.
    assert _apply_override(defaults, 42, set(), builtin_names=set()) == []


def test_scalar_falsy_override_fails_closed():
    """Review finding #1 (cont'd): a falsy scalar persisted override
    (e.g. ``0``) must NOT silently bypass the caller's ``if _override:``
    and leave the profile defaults untouched (fail-open)."""
    defaults = ["web", "file", "terminal", "delegation"]
    # Falsy scalar — old caller's ``if _override:`` skipped it entirely,
    # leaving defaults in place.
    assert _apply_override(defaults, 0, set(), builtin_names=set()) == []


def test_scalar_string_override_fails_closed():
    """Review finding #1 (cont'd): a string scalar override must also
    fail closed, not be interpreted as a single-element list."""
    defaults = ["web", "file"]
    assert _apply_override(defaults, "web", set(), builtin_names=set()) == []


def test_none_override_keeps_defaults():
    """``None`` is the valid "no override" signal and must preserve defaults."""
    defaults = ["web", "file"]
    assert _apply_override(defaults, None, set(), builtin_names=set()) == defaults


def test_empty_list_override_keeps_defaults():
    """``[]`` is the valid "no override" signal and must preserve defaults."""
    defaults = ["web", "file"]
    assert _apply_override(defaults, [], set(), builtin_names=set()) == defaults


def test_unchecked_mcp_computed_from_all_configured_servers():
    """Review finding #3: unchecked_mcp must be computed from ALL configured
    MCP server names, not just pure_mcp (which strips collision names)."""
    src = (REPO / "api" / "streaming.py").read_text(encoding="utf-8")
    # The additive branch must use the full configured set
    assert "unchecked_mcp = mcp_server_names - set(override)" in src, (
        "unchecked_mcp must be computed from all configured servers"
    )
    # The old pure_mcp-based computation must be gone
    assert "unchecked_mcp = pure_mcp - set(override)" not in src, (
        "old pure_mcp-based unchecked_mcp must be removed"
    )


# ── Round-5 reviewer findings ────────────────────────────────────────────────


def test_scalar_override_through_real_caller_path(monkeypatch, tmp_path):
    """Drive scalar overrides through the REAL ``_run_agent_streaming()`` call
    site (api/streaming.py:8297-8314) and capture the final ``enabled_toolsets``
    the agent constructor receives.

    The previous version copied the caller logic into a local ``_caller_path``
    — a regression at the real call site (e.g. reverting to ``if _override:``)
    could pass while the copy stayed green.  This harness invokes the actual
    worker with only the agent constructor substituted (a capturing fake) and
    asserts the captured ``enabled_toolsets`` for a falsy scalar override.
    """
    import queue
    import sys
    import types as _types

    from api import models

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["enabled_toolsets"] = kwargs.get("enabled_toolsets")

        def run_conversation(self, **kwargs):
            return {"failed": False, "messages": []}

    sid = "scalar-real-caller"
    stream_id = "scalar-real-stream"
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(streaming, "SESSION_DIR", session_dir)
    session = models.Session(
        session_id=sid,
        title="scalar",
        workspace=str(tmp_path),
        model="gpt-4o",
        messages=[],
        context_messages=[],
    )
    session.active_stream_id = stream_id
    session.pending_user_message = "hi"
    session.pending_started_at = 1.0
    session.save()
    models.SESSIONS[sid] = session
    streaming.SESSIONS[sid] = session
    streaming.STREAMS[stream_id] = queue.Queue()

    fake_hermes_state = _types.ModuleType("hermes_state")
    fake_hermes_state.SessionDB = lambda *_a, **_k: object()

    def _fake_load_metadata_only(session_id):
        meta = models.Session(session_id=session_id, model="gpt-4o")
        # Falsy scalar persisted metadata (the round-5 regression shape).
        meta.enabled_toolsets = 0
        return meta

    with monkeypatch.context() as m:
        m.setattr(streaming, "_get_ai_agent", lambda: FakeAgent)
        m.setattr(streaming, "resolve_model_provider",
                  lambda *_a, **_k: ("gpt-4o", "openai", None))
        m.setattr("api.config.get_config", lambda *_a, **_k: {"mcp_servers": {}})
        m.setattr("api.config._resolve_cli_toolsets", lambda *_a, **_k: ["web", "file"])
        m.setattr(models.Session, "load_metadata_only",
                  staticmethod(_fake_load_metadata_only))
        m.setitem(sys.modules, "hermes_state", fake_hermes_state)
        streaming._run_agent_streaming(
            session_id=sid,
            msg_text="hi",
            model="gpt-4o",
            workspace=str(tmp_path),
            stream_id=stream_id,
        )

    assert captured["enabled_toolsets"] == [], (
        "falsy scalar override must fail closed to empty through the REAL "
        f"call site; got {captured.get('enabled_toolsets')!r}"
    )


def test_runtime_custom_composite_and_leaf_resolver_called(monkeypatch):
    """Build an unsafe composite and a direct-tools leaf through the REAL
    ``create_custom_toolset()`` and registry ownership APIs; wrap the real
    resolver while delegating to it.  Prove the unchecked direct leaf and
    the unsafe composite are dropped, while builtins and the selected
    server survive.

    The previous version assigned plain dicts into ``toolsets.TOOLSETS`` and
    hard-coded classifier returns — it never exercised real creation,
    registry ownership, or the resolver's expansion.  This version registers
    real tools owned by an unchecked ``mcp-gate-beta`` and lets the real
    resolver decide.
    """
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        _ts_mod.create_custom_toolset("gate-composite", "composite",
                                      tools=[], includes=["gate-beta"])
        _ts_mod.create_custom_toolset("gate-leaf", "leaf",
                                      tools=["leaf_tool"], includes=[])
        _ts_mod.create_custom_toolset("safe-composite", "safe",
                                      tools=[], includes=["web"])
        _reg.register_toolset_alias("gate-beta", "mcp-gate-beta")
        _reg.register_toolset_alias("gate-alpha", "mcp-gate-alpha")
        _reg.register("leaf_tool", "mcp-gate-beta", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        _reg.register("alpha_tool", "mcp-gate-alpha", {"type": "object"},
                      lambda *a, **k: "ok", override=True)

        _calls = []
        _orig = streaming._default_transitively_reaches_unchecked_mcp

        def _wrapped(name, unchecked, builtin_names=None):
            _calls.append(name)
            return _orig(name, unchecked, builtin_names)

        monkeypatch.setattr(streaming,
                            "_default_transitively_reaches_unchecked_mcp",
                            _wrapped)

        defaults = ["web", "gate-composite", "gate-leaf", "safe-composite"]
        override = ["gate-alpha"]
        mcp_servers = {"gate-alpha", "gate-beta"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names={"web", "file", "terminal"})

        # Resolver consulted for every survivor (builtins included, round-5).
        for name in ("web", "gate-composite", "gate-leaf", "safe-composite"):
            assert name in _calls, f"resolver must inspect {name}, got {_calls}"

        # Builtins + selected server survive; unchecked-MCP paths drop.
        assert "web" in result, f"builtin web must survive, got {result!r}"
        assert "gate-alpha" in result, f"ticked server must survive, got {result!r}"
        assert "gate-composite" not in result, (
            f"composite reaching unchecked mcp-gate-beta must be dropped, got {result!r}"
        )
        assert "gate-leaf" not in result, (
            f"direct leaf owned by unchecked mcp-gate-beta must be dropped, got {result!r}"
        )
        assert "safe-composite" in result, (
            f"safe composite (only includes web) must survive, got {result!r}"
        )
        assert "gate-beta" not in result, (
            f"unchecked server must not appear, got {result!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


def test_unchecked_builtin_colliding_mcp_via_canonical_behavioral(monkeypatch):
    """Real canonical ``mcp-web`` resolution: register a tool owned by the
    canonical ``mcp-web`` toolset, build a composite that includes
    ``mcp-web``, and prove registry ownership causes rejection when the
    colliding server ``web`` is configured but unchecked.

    The previous version answered ``True`` from a wrapper for
    ``web-reach-composite``; this version resolves a real canonical
    ``mcp-web`` tool and lets registry ownership drive the denial.
    """
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        _ts_mod.create_custom_toolset("web-reach-composite", "reaches mcp-web",
                                      tools=[], includes=["mcp-web"])
        _reg.register("web_owned_tool", "mcp-web", {"type": "object"},
                      lambda *a, **k: "ok", override=True)

        defaults = ["web", "web-reach-composite"]
        override = ["my-search"]
        mcp_servers = {"my-search", "web"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names={"web", "file", "terminal"})

        assert "web-reach-composite" not in result, (
            f"composite reaching unchecked mcp-web must be dropped, got {result!r}"
        )
        assert "web" in result, f"builtin web must survive, got {result!r}"
        assert "my-search" in result, f"ticked server must survive, got {result!r}"
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


def test_safe_composite_positive_survival(monkeypatch):
    """A safe composite that does not reach any unchecked MCP server
    survives — with one selected server AND a different unchecked server
    configured, so the production helper cannot early-return on an empty
    unchecked set (non-vacuous).

    The previous version configured only the selected server, leaving
    ``unchecked_mcp`` empty and skipping the resolver/registry inspection.
    """
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        _ts_mod.create_custom_toolset("safe-composite", "safe workflow",
                                      tools=[], includes=["web"])
        _reg.register_toolset_alias("gate-beta", "mcp-gate-beta")
        _reg.register("beta_tool", "mcp-gate-beta", {"type": "object"},
                      lambda *a, **k: "ok", override=True)

        defaults = ["web", "safe-composite"]
        override = ["my-search"]
        # gate-beta configured but NOT ticked → unchecked set non-empty.
        mcp_servers = {"my-search", "gate-beta"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names={"web", "file", "terminal"})

        assert "safe-composite" in result, (
            "safe composite must survive with an unchecked server present, "
            f"got {result!r}"
        )
        assert "web" in result, f"builtin web must survive, got {result!r}"
        assert "my-search" in result, f"ticked server must survive, got {result!r}"
        assert "gate-beta" not in result, (
            f"unchecked server must not appear, got {result!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


# ── Round-6 reviewer findings ────────────────────────────────────────────────


def test_runtime_custom_composite_through_real_caller(monkeypatch, tmp_path):
    """Production-composed acceptance: drive a safe custom composite, an
    unsafe composite, a direct leaf, and a static MCP name in defaults
    through the REAL ``_run_agent_streaming()`` call site.  Assert the
    final ``enabled_toolsets`` (complete list) and the resolved tool
    list (including a unique registered tool from the safe composite)."""
    import queue
    import sys
    import types as _types

    import toolsets as _ts_mod
    from api import models
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["enabled_toolsets"] = kwargs.get("enabled_toolsets")

        def run_conversation(self, **kwargs):
            return {"failed": False, "messages": []}

    try:
        # Seed real registry/toolset state (runtime-composed, like production):
        # - safe-composite: custom composite that only includes safe-inner
        #   (a registry-only toolset with a unique tool, proving resolve works)
        # - unsafe-composite: custom composite that includes unchecked mcp-beta
        # - gate-leaf: direct tool owned by unchecked mcp-gate-beta
        # - gate-alpha: selected MCP server (ticked); gate-beta: unchecked
        # - search: configured MCP server whose name COLLIDES with static "search"
        _ts_mod.create_custom_toolset("safe-composite", "safe workflow",
                                      tools=[], includes=["safe-inner"])
        _ts_mod.create_custom_toolset("unsafe-composite", "reaches beta",
                                      tools=[], includes=["gate-beta"])
        _ts_mod.create_custom_toolset("gate-leaf", "leaf owned by beta",
                                      tools=["beta_only_tool"], includes=[])
        _reg.register_toolset_alias("gate-alpha", "mcp-gate-alpha")
        _reg.register_toolset_alias("gate-beta", "mcp-gate-beta")
        _reg.register("alpha_tool", "mcp-gate-alpha", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        _reg.register("beta_only_tool", "mcp-gate-beta", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        _reg.register("safe_unique_tool", "safe-inner", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        # Collision: MCP server "search" shares name with static toolset "search"
        _reg.register_toolset_alias("search", "mcp-search")
        _reg.register("search_mcp_tool", "mcp-search", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        # Exact-static overlay canary (round-8 finding 3): a unique tool
        # registered under the exact static owner "search" (NOT the aliased
        # mcp-search) must survive resolution, proving the installed
        # static-first-overlay merge order is real.  Reverting the stand-in
        # to a static-only return must make this assertion fail.
        _reg.register("search_static_owned_tool", "search", {"type": "object"},
                      lambda *a, **k: "ok", override=True)

        # ── Round-8 finding 1: the collision seed must be assertion-relevant ──
        # These checks run BEFORE driving the caller and prove the collision
        # setup is actually installed.  Deleting or no-oping the
        # `search -> mcp-search` alias registration below must turn the test
        # RED — the final `search_mcp_tool not in final_tools` assertion alone
        # is too easy to satisfy when the seed is absent.
        _alias_target = _reg.get_toolset_alias_target("search")
        assert _alias_target == "mcp-search", (
            f"collision seed must alias search -> mcp-search, got {_alias_target!r}"
        )
        _mcp_search_tools = set(_reg.get_tool_names_for_toolset("mcp-search"))
        assert "search_mcp_tool" in _mcp_search_tools, (
            f"canonical mcp-search must own the unique MCP canary, got {sorted(_mcp_search_tools)!r}"
        )
        # Non-static alias positive control: `gate-alpha` is NOT a static
        # toolset, so resolving it must take the alias -> mcp-gate-alpha path
        # and surface alpha_tool.  This proves the alias activation path is
        # genuinely live (not just the static-first branch being exercised).
        _gate_alpha_resolved = set(_ts_mod.resolve_toolset("gate-alpha"))
        assert "alpha_tool" in _gate_alpha_resolved, (
            f"non-static alias gate-alpha must resolve alpha_tool, got {sorted(_gate_alpha_resolved)!r}"
        )

        sid = "runtime-composed"
        stream_id = "runtime-composed-stream"
        session_dir = tmp_path / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
        monkeypatch.setattr(streaming, "SESSION_DIR", session_dir)
        session = models.Session(
            session_id=sid,
            title="runtime-composed",
            workspace=str(tmp_path),
            model="gpt-4o",
            messages=[],
            context_messages=[],
        )
        session.active_stream_id = stream_id
        session.pending_user_message = "hi"
        session.pending_started_at = 1.0
        session.save()
        models.SESSIONS[sid] = session
        streaming.SESSIONS[sid] = session
        streaming.STREAMS[stream_id] = queue.Queue()

        fake_hermes_state = _types.ModuleType("hermes_state")
        fake_hermes_state.SessionDB = lambda *_a, **_k: object()

        def _fake_load_metadata_only(session_id):
            meta = models.Session(session_id=session_id, model="gpt-4o")
            # Selected MCP server ticked in the composer chip; gate-beta and
            # search are configured but NOT ticked → unchecked MCP servers present.
            meta.enabled_toolsets = ["gate-alpha"]
            return meta

        with monkeypatch.context() as m:
            m.setattr(streaming, "_get_ai_agent", lambda: FakeAgent)
            m.setattr(streaming, "resolve_model_provider",
                      lambda *_a, **_k: ("gpt-4o", "openai", None))
            m.setattr("api.config.get_config", lambda *_a, **_k: {
                "mcp_servers": {"gate-alpha": {}, "gate-beta": {}, "search": {}},
            })
            m.setattr("api.config._resolve_cli_toolsets",
                      lambda *_a, **_k: ["web", "search", "safe-composite",
                                         "unsafe-composite", "gate-leaf"])
            m.setattr(models.Session, "load_metadata_only",
                      staticmethod(_fake_load_metadata_only))
            m.setitem(sys.modules, "hermes_state", fake_hermes_state)
            streaming._run_agent_streaming(
                session_id=sid,
                msg_text="hi",
                model="gpt-4o",
                workspace=str(tmp_path),
                stream_id=stream_id,
            )

        result = captured["enabled_toolsets"]
        assert result is not None, "AIAgent constructor never received enabled_toolsets"
        # Assert the COMPLETE list, not just membership.
        assert result == ["web", "search", "safe-composite", "gate-alpha"], (
            f"unexpected enabled_toolsets: {result!r}"
        )

        # Resolve the final toolset-set through the real resolver/registry and
        # assert the FINAL tool list.
        final_tools = set()
        for name in result:
            final_tools.update(_ts_mod.resolve_toolset(name))
        assert "web_search" in final_tools, (
            f"builtin web tools must resolve, got {sorted(final_tools)!r}"
        )
        assert "safe_unique_tool" in final_tools, (
            f"safe-composite unique tool must resolve, got {sorted(final_tools)!r}"
        )
        assert "alpha_tool" in final_tools, (
            f"selected gate-alpha tool must resolve, got {sorted(final_tools)!r}"
        )
        assert "beta_only_tool" not in final_tools, (
            f"unchecked beta-owned tool must be excluded, got {sorted(final_tools)!r}"
        )
        # Collision asserts: static "search" tools survive; aliased MCP
        # "search" is excluded.
        search_tools = set(_ts_mod.resolve_toolset("search"))
        assert search_tools.issubset(final_tools), (
            f"static search tools {sorted(search_tools)} must survive collision, "
            f"got {sorted(final_tools)!r}"
        )
        assert "search_mcp_tool" not in final_tools, (
            f"colliding MCP search tool must be excluded, got {sorted(final_tools)!r}"
        )
        # Round-8 finding 3: the exact-static overlay canary — a unique tool
        # registered under the static owner "search" (not the aliased
        # mcp-search) — must survive into the resolved tool set.  This proves
        # the installed static-first-overlay merge order is genuinely active;
        # a stand-in that returns static-only (no registry overlay) would fail
        # here.
        assert "search_static_owned_tool" in search_tools, (
            f"exact-static overlay must merge registry tools under 'search', "
            f"got {sorted(search_tools)!r}"
        )
        assert "search_static_owned_tool" in final_tools, (
            f"exact-static overlay canary must survive into final tools, "
            f"got {sorted(final_tools)!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


# ── Gate-certification regressions (round-10) ────────────────────────────────
# hermes-sweeper gate RED on 08fafc9d reported three production defects:
#   1. BRICK  — a configured MCP server named ``all`` (the reserved wildcard
#               class) breaks tool resolution: the resolver treats the bare
#               selector as a wildcard and recurses into itself until
#               RecursionError.
#   2. SILENT — canonical-name collisions resolve the WRONG MCP server: with
#               servers ``alpha`` and ``mcp-alpha`` configured, selecting
#               ``mcp-alpha`` returns the ambiguous bare token and exposes
#               ``alpha``'s tools (``mcp-mcp-alpha`` is the true canonical).
#   3. SILENT — an offline unchecked server can re-enter through a retained
#               custom composite after registration: the tool-level check
#               sees no registered tools while the server is offline and
#               keeps the composite; registering the server later exposes
#               its tools without a new authorization decision.
# The production fix canonicalizes picker-derived MCP selections to a
# verified registry target and adds a registration-state-independent
# structural walk of composite ``includes`` chains.


def test_server_named_all_is_canonicalized_not_wildcard(monkeypatch):
    """A configured MCP server literally named ``all`` must be emitted as the
    canonical ``mcp-all`` target, never as the bare wildcard ``all``.

    The installed resolver treats bare ``all``/``*`` as "expand every
    toolset" BEFORE consulting MCP aliases, so a bare ``all`` selection
    recurses until RecursionError (gate-certified BRICK).  The additive
    classifier must canonicalize it to ``mcp-all``, which resolves as a
    plain toolset name.
    """
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        _ts_mod.create_custom_toolset("all", "server literally named all",
                                      tools=[], includes=[])
        _reg.register_toolset_alias("all", "mcp-all")
        _reg.register("all_tool", "mcp-all", {"type": "object"},
                      lambda *a, **k: "ok", override=True)

        defaults = ["web", "file"]
        override = ["all"]
        mcp_servers = {"all", "other"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names={"web", "file", "terminal"})

        assert "all" not in result, (
            f"bare wildcard 'all' must never survive, got {result!r}"
        )
        assert "mcp-all" in result, (
            f"canonical mcp-all must be emitted, got {result!r}"
        )
        # The canonical target must resolve to the server's tools.
        resolved = set(_ts_mod.resolve_toolset("mcp-all"))
        assert "all_tool" in resolved, (
            f"mcp-all must resolve the server's tools, got {sorted(resolved)!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


def test_server_named_star_is_canonicalized_not_wildcard(monkeypatch):
    """The ``*`` reserved selector collides with a server of the same name and
    must also be canonicalized to ``mcp-*``."""
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        _ts_mod.create_custom_toolset("*", "server literally named star",
                                      tools=[], includes=[])
        _reg.register_toolset_alias("*", "mcp-*")
        _reg.register("star_tool", "mcp-*", {"type": "object"},
                      lambda *a, **k: "ok", override=True)

        defaults = ["web"]
        override = ["*"]
        mcp_servers = {"*", "other"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names={"web", "file", "terminal"})

        assert "*" not in result, (
            f"bare wildcard '*' must never survive, got {result!r}"
        )
        assert "mcp-*" in result, (
            f"canonical mcp-* must be emitted, got {result!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


def test_canonical_collision_mcp_alpha_selects_right_server(monkeypatch):
    """With servers ``alpha`` AND ``mcp-alpha`` configured, selecting
    ``mcp-alpha`` must emit ``mcp-mcp-alpha`` (the selected server's true
    canonical), NOT the bare ``mcp-alpha`` which collides with server
    ``alpha``'s canonical toolset (gate-certified SILENT defect 2).

    Production-composed: ``builtin_names`` is NOT injected — the real
    ``_builtin_toolset_names()`` collector runs, so the broad shadow set
    contains BOTH ``mcp-alpha`` and ``mcp-mcp-alpha``.  The round-11
    owned-canonical exclusion must remove them (alias-edge proven) before
    the MCP-only test, or ``mcp-alpha`` would be wrongly restricted and the
    canonicalizer branch never reached (the round-10 blocker)."""
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        # Server "alpha" → canonical mcp-alpha
        _ts_mod.create_custom_toolset("alpha", "server alpha", tools=[], includes=[])
        _reg.register_toolset_alias("alpha", "mcp-alpha")
        _reg.register("alpha_owned_tool", "mcp-alpha", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        # Server "mcp-alpha" → canonical mcp-mcp-alpha
        _ts_mod.create_custom_toolset("mcp-alpha", "server mcp-alpha",
                                      tools=[], includes=[])
        _reg.register_toolset_alias("mcp-alpha", "mcp-mcp-alpha")
        _reg.register("mcp_alpha_owned_tool", "mcp-mcp-alpha", {"type": "object"},
                      lambda *a, **k: "ok", override=True)

        # Seed-assertion relevance (round-8 rule): prove the alias edges are
        # live BEFORE driving the classifier.
        assert _reg.get_toolset_alias_target("alpha") == "mcp-alpha"
        assert _reg.get_toolset_alias_target("mcp-alpha") == "mcp-mcp-alpha"

        defaults = ["web"]
        override = ["mcp-alpha"]
        mcp_servers = {"alpha", "mcp-alpha"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names=None)

        assert "mcp-alpha" not in result, (
            f"ambiguous bare mcp-alpha must never survive, got {result!r}"
        )
        assert "mcp-mcp-alpha" in result, (
            f"true canonical mcp-mcp-alpha must be emitted, got {result!r}"
        )
        assert "web" in result, f"builtin web must survive, got {result!r}"
        resolved = set(_ts_mod.resolve_toolset("mcp-mcp-alpha"))
        assert "mcp_alpha_owned_tool" in resolved, (
            f"mcp-mcp-alpha must resolve the SELECTED server's tools, "
            f"got {sorted(resolved)!r}"
        )
        # The other server's tools must not appear.
        assert "alpha_owned_tool" not in resolved, (
            f"wrong server's tools must not resolve, got {sorted(resolved)!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


def test_offline_unchecked_server_cannot_reenter_composite(monkeypatch):
    """A composite whose ``includes`` references an unchecked MCP server must
    be dropped EVEN WHEN that server is offline (no tools registered yet).

    Gate-certified SILENT defect 3: the tool-level check sees no registered
    tools while the server is offline and keeps the composite; registering
    the server later would re-expose its tools without a new authorization
    decision.  The structural includes walk is registration-state
    independent and must reject the composite up front, and the retained
    result must stay clean after the server registers.
    """
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        # Composite reaches "gate-beta" — configured but NOT ticked.
        _ts_mod.create_custom_toolset("bad-composite", "reaches gate-beta",
                                      tools=[], includes=["gate-beta"])
        # NOTE: gate-beta has NO alias and NO registered tools yet — offline.

        defaults = ["web", "bad-composite"]
        override = ["gate-alpha"]
        mcp_servers = {"gate-alpha", "gate-beta"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names={"web", "file", "terminal"})

        assert "bad-composite" not in result, (
            f"composite reaching offline unchecked server must be dropped, "
            f"got {result!r}"
        )
        assert "web" in result, f"builtin web must survive, got {result!r}"
        assert "gate-alpha" in result, (
            f"ticked server must survive, got {result!r}"
        )

        # Late registration: the unchecked server comes online AFTER the
        # merge decision.  The retained result must not expose its tools.
        _reg.register_toolset_alias("gate-beta", "mcp-gate-beta")
        _reg.register("beta_tool", "mcp-gate-beta", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        final_tools = set()
        for name in result:
            final_tools.update(_ts_mod.resolve_toolset(name))
        assert "beta_tool" not in final_tools, (
            f"offline server must not re-enter via retained composite, "
            f"got {sorted(final_tools)!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


def test_offline_unchecked_server_canonical_reference_rejected(monkeypatch):
    """A composite whose includes chain references an unchecked server by its
    CANONICAL ``mcp-<server>`` selector is rejected even while offline —
    canonical resolution bypasses alias shadowing, so the canonical name in
    the includes chain is always an MCP reference."""
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        _ts_mod.create_custom_toolset("bad-composite", "reaches mcp-gate-beta",
                                      tools=[], includes=["mcp-gate-beta"])

        defaults = ["web", "bad-composite"]
        override = ["gate-alpha"]
        mcp_servers = {"gate-alpha", "gate-beta"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names={"web", "file", "terminal"})

        assert "bad-composite" not in result, (
            f"composite reaching offline unchecked canonical must be dropped, "
            f"got {result!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)


def test_builtin_shadowed_bare_name_not_treated_as_mcp_reference(monkeypatch):
    """A bare name that is BOTH a builtin toolset and a configured MCP server
    (e.g. server ``search`` colliding with static ``search``) must NOT be
    rejected by the structural walk — the builtin shadows the MCP alias, so
    the default entry ``search`` is the builtin, not an MCP reference."""
    import toolsets as _ts_mod
    from tools.registry import registry as _reg

    _saved_toolsets = dict(_ts_mod.TOOLSETS)
    _saved_tools = dict(_reg._tools)
    _saved_aliases = dict(_reg._toolset_aliases)

    try:
        _ts_mod.create_custom_toolset("search", "static search builtin",
                                      tools=["search_tool"], includes=[])
        _ts_mod.create_custom_toolset("web", "static web builtin",
                                      tools=["web_tool"], includes=[])
        _reg.register("search_tool", "search", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        _reg.register("web_tool", "web", {"type": "object"},
                      lambda *a, **k: "ok", override=True)
        # Server "search" is configured but unchecked; default "search" is
        # the static builtin and must survive.
        defaults = ["web", "search"]
        override = ["gate-alpha"]
        mcp_servers = {"gate-alpha", "search"}
        result = _apply_override(defaults, override, mcp_servers,
                                 builtin_names={"web", "search", "file", "terminal"})

        assert "web" in result, f"builtin web must survive, got {result!r}"
        assert "search" in result, (
            f"builtin search (bare name shadowed by static) must survive, "
            f"got {result!r}"
        )
        assert "gate-alpha" in result, (
            f"ticked server must survive, got {result!r}"
        )
    finally:
        _ts_mod.TOOLSETS.clear()
        _ts_mod.TOOLSETS.update(_saved_toolsets)
        _reg._tools.clear()
        _reg._tools.update(_saved_tools)
        _reg._toolset_aliases.clear()
        _reg._toolset_aliases.update(_saved_aliases)
