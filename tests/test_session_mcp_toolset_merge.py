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

from pathlib import Path

from api.streaming import _apply_session_toolset_override as _apply_override

REPO = Path(__file__).resolve().parents[1]


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
    (static TOOLSETS keys + registered names), or ``None`` if the registry is
    unavailable in this environment (fail-closed)."""
    import api.streaming as streaming

    try:
        from tools.registry import registry as _reg
    except ImportError:
        return None  # CI without tools.registry — fail-closed

    _orig = _reg.get_registered_toolset_names
    monkeypatch.setattr(_reg, "get_registered_toolset_names",
                        lambda: list(registered_names))
    try:
        return streaming._builtin_toolset_names()
    finally:
        monkeypatch.setattr(_reg, "get_registered_toolset_names", _orig)


def test_mcp_canonical_name_collision_restricts(monkeypatch):
    """When MCP servers ``alpha`` and ``mcp-alpha`` coexist, the bare token
    ``mcp-alpha`` (submitted by the picker for the second server) collides with
    the *canonical* name ``mcp-alpha`` (owned by the first server).  The
    override must restrict — not flip additive, which would restore defaults
    and let the runtime resolver match server ``alpha`` instead of the
    intended ``mcp-alpha``.

    The shadow set is obtained from the real ``_builtin_toolset_names()``
    (monkeypatched registry), not hand-constructed, so the test exercises the
    actual enumeration path and would fail if the old ``mcp-*`` filtering were
    restored."""
    builtin = _patch_registry_and_get_builtin_names(
        monkeypatch, ["mcp-alpha", "mcp-mcp-alpha"])

    if builtin is None:
        # Registry unavailable → fail-closed RESTRICT is correct.
        result = _apply_override(
            ["web", "file", "terminal", "delegation"], ["mcp-alpha"],
            {"alpha", "mcp-alpha"}, builtin_names=None)
        assert result == ["mcp-alpha"], (
            "unavailable registry must restrict on mcp-* canonical collision "
            "(got {})".format(result)
        )
        return

    # Canonical ``mcp-alpha`` (from server ``alpha``) must be in the shadow
    # set so the bare token ``mcp-alpha`` (from server ``mcp-alpha``) collides.
    assert "mcp-alpha" in builtin, (
        "canonical 'mcp-alpha' must be in the shadow set (got {})".format(builtin)
    )
    assert "mcp-mcp-alpha" in builtin, (
        "canonical 'mcp-mcp-alpha' must be in the shadow set"
    )

    result = _apply_override(
        ["web", "file", "terminal", "delegation"], ["mcp-alpha"],
        {"alpha", "mcp-alpha"}, builtin_names=builtin)

    assert result == ["mcp-alpha"], (
        "bare token 'mcp-alpha' collides with canonical 'mcp-alpha' from "
        "server 'alpha' — must RESTRICT, not flip additive (got {})".format(result)
    )


def test_ordinary_server_stays_additive_with_canonical_in_shadow(monkeypatch):
    """Negative control: a normal server ``foo`` (pick token ``foo``) stays
    additive even though its canonical name ``mcp-foo`` is in the shadow set.
    The bare token ``foo`` ≠ ``mcp-foo``, so there is no collision.

    Uses the real ``_builtin_toolset_names()`` (monkeypatched registry), not
    hand-constructed shadow set."""
    builtin = _patch_registry_and_get_builtin_names(
        monkeypatch, ["mcp-foo", "mcp-bar"])

    if builtin is None:
        # Registry unavailable → fail-closed RESTRICT.  We cannot prove
        # ``foo`` is a bare MCP tick without the registry, so RESTRICT is
        # the safe fallback.  The additive path is verified below when the
        # registry is available.
        result = _apply_override(
            ["web", "file"], ["foo"], {"foo", "bar"}, builtin_names=None)
        assert result == ["foo"], (
            "unavailable registry must fail closed to RESTRICT (got {})".format(result)
        )
        return

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

    Uses the real ``_builtin_toolset_names()`` (monkeypatched registry), not
    hand-constructed shadow set."""
    builtin = _patch_registry_and_get_builtin_names(
        monkeypatch, ["mcp-browser"])

    if builtin is None:
        result = _apply_override(
            ["web", "file"], ["mcp-browser"], {"mcp-browser"},
            builtin_names=None)
        assert result == ["mcp-browser"], (
            "unavailable registry must restrict on mcp-* plugin collision "
            "(got {})".format(result)
        )
        return

    assert "mcp-browser" in builtin, (
        "plugin canonical 'mcp-browser' must be in the shadow set "
        "(got {})".format(builtin)
    )

    result = _apply_override(
        ["web", "file"], ["mcp-browser"], {"mcp-browser"},
        builtin_names=builtin)

    assert result == ["mcp-browser"], (
        "plugin canonical 'mcp-browser' colliding with same-named MCP server "
        "must RESTRICT (got {})".format(result)
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
