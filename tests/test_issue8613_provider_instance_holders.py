"""#8613: an evicted agent's memory provider may only be released by its LAST holder.

WebUI caches AIAgent instances so a memory provider can carry state across turns.
When an agent is evicted, ``api.streaming._close_evicted_agent_at_session_boundary``
commits pending memory and then calls ``shutdown_memory_provider``. When the
session's owner profile cannot be resolved, ``call_as_session_owner`` refuses that
call — session end is a commit, and an unknown owner never commits — which also
skipped the provider teardown inside it, so the provider's ``mem-sync`` writer and
its clients (Hindsight's aiohttp session) lingered until the WebUI process exited.

Releasing the provider is only safe per INSTANCE, not per manager: a provider that
hands every agent one process-wide instance (mnemosyne-hermes 0.7.1) is shared by
every agent of the profile that claimed it — the #8517 P4 guard stops a second
PROFILE from rebinding it, not a second agent of the same one. Each agent owns its
own MemoryManager, and a manager-level teardown (``claim_teardown`` /
``release_soft`` / ``shutdown_all``) shuts the shared instance down under the
agents still using it. That is why the first version of this fix was reverted
(#8636). WebUI therefore tracks which agents hold each provider instance and
releases only when the agent reaching the boundary is the last holder.

These tests drive the REAL ``MemoryManager`` (``release_soft`` -> ``claim_teardown``
-> bounded flush -> ``shutdown_all``) with a shared-instance provider, and build
every agent through the REAL ``construct_profile_scoped_agent`` so the holder table
is populated exactly the way a turn populates it. Only the Agent class is replaced,
by a stand-in that binds memory the way ``MemoryManager.initialize_for_agent`` does
and ports ``run_agent.AIAgent.shutdown_memory_provider``.

"Provider worker": the provider-owned client a background writer uses. ``shutdown()``
closes it; ``remember()`` is the save that then fails.

"Unowned session": one whose owner profile cannot be resolved — no recorded owner
and no session record, so ``session_owner_profile_home`` answers ``""``. The
lifecycle entry is clean (no uncommitted generation), which is the only shape in
which the eviction path reaches the provider shutdown at all.
"""
# ruff: noqa: F811  (pytest fixtures imported from the #8517 P0 module)
from __future__ import annotations

import logging
import threading

import pytest

from tests.test_issue8517_profile_scoped_memory import (  # noqa: F401  (fixtures)
    _key,
    hc,
    homes,
)

import api.memory_profile_scope as scope
import api.session_lifecycle as lifecycle
import api.streaming as streaming

mm_mod = pytest.importorskip("agent.memory_manager")
if not callable(getattr(getattr(mm_mod, "MemoryManager", None), "release_soft", None)):
    pytest.skip("installed Agent predates MemoryManager.release_soft", allow_module_level=True)
from agent.memory_provider import MemoryProvider  # noqa: E402

PROVIDER_NAME = "8613mem"


# ── the stand-in provider: one instance, shared by every agent of the profile ──

class ProviderClient:
    """The provider-owned worker the issue is about."""

    def __init__(self):
        self.closed = False
        self.writes: list[str] = []

    def close(self):
        self.closed = True


class SharedInstanceProvider(MemoryProvider):
    """mnemosyne-hermes 0.7.1 shape: ``register()`` hands every agent the SAME
    instance, so every agent of the profile that claimed it shares one client."""

    name = PROVIDER_NAME

    def __init__(self):
        self.client = ProviderClient()
        self.inits: list[dict] = []
        self.shutdowns = 0

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.inits.append({"session": session_id, "home": str(kwargs.get("hermes_home") or "")})

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        self.shutdowns += 1
        self.client.close()

    def remember(self, marker):
        if self.client.closed:
            raise RuntimeError("memory provider client is closed: the save failed")
        self.client.writes.append(marker)


def _fresh_instance():
    """The documented provider contract: a NEW instance per agent."""
    return SharedInstanceProvider()


# ── the stand-in Agent ───────────────────────────────────────────────────────

def make_holder_agent_cls(provider_for_build, *, home, identity="alpha"):
    """Stand-in AIAgent. ``__init__`` binds memory the way the real Agent does
    (``MemoryManager.initialize_for_agent`` -> ``claim_instance`` -> ``initialize``);
    ``shutdown_memory_provider`` is ``run_agent.AIAgent.shutdown_memory_provider``
    (claim teardown -> ``on_session_end`` -> ``shutdown_all``); ``save`` is one
    memory write through the agent's provider."""

    class HolderAgent:
        built: list = []

        def __init__(self, session_id=None, skip_memory=False, **kwargs):
            self.session_id = session_id
            self._memory_manager = None
            self._memory_provider_shutdown = False
            self._session_messages = []
            if not skip_memory:
                manager = mm_mod.MemoryManager()
                manager.add_provider(provider_for_build())
                manager.initialize_for_agent(
                    session_id=session_id, hermes_home=str(home), agent_identity=identity
                )
                self._memory_manager = manager
            HolderAgent.built.append(self)

        def shutdown_memory_provider(self, messages=None):
            if self._memory_provider_shutdown:
                return
            self._memory_provider_shutdown = True
            if self._memory_manager and self._memory_manager.claim_teardown(session_end=True):
                try:
                    self._memory_manager.on_session_end(messages or [])
                except Exception:  # pragma: no cover - the real Agent logs and continues
                    pass
                self._memory_manager.shutdown_all()

        def save(self, marker):
            """One memory save through the agent's provider (tool call / autosave)."""
            provider = self._memory_manager.get_provider(PROVIDER_NAME)
            assert provider is not None, "the agent has no external memory provider"
            provider.remember(marker)

    return HolderAgent


def _build(home, provider_for_build, session_id, *, identity="alpha"):
    """One agent built through the production constructor, so the holder table is
    populated exactly the way a real turn populates it."""
    agent, cacheable = scope.construct_profile_scoped_agent(
        make_holder_agent_cls(provider_for_build, home=home, identity=identity),
        {"model": "m"},
        profile_home=str(home),
        session_id=session_id,
    )
    assert cacheable is True, "a memory-bearing build must stay cacheable"
    assert agent._memory_manager is not None
    return agent


def _shared_agent(home, provider, session_id):
    return _build(home, lambda: provider, session_id)


@pytest.fixture
def holder_registry(monkeypatch):
    """Every test starts from the Agent's own ``claim_instance`` and from empty
    WebUI registries (the #8517 P4 ownership table and the #8613 holder table);
    both are restored afterwards."""
    original = mm_mod.claim_instance
    while getattr(original, "__wrapped__", None) is not None:
        original = original.__wrapped__
    monkeypatch.setattr(mm_mod, "claim_instance", original)
    monkeypatch.setattr(scope, "_logged_once", set())
    if hasattr(scope, "_instance_owners"):
        monkeypatch.setattr(scope, "_instance_owners", {})
    if hasattr(scope, "_provider_holders"):
        monkeypatch.setattr(scope, "_provider_holders", {})
    lifecycle._reset_for_tests()
    yield scope
    lifecycle._reset_for_tests()


# ── (a) the last holder may release: the leak #8613 reports ──────────────────

def test_unowned_eviction_releases_the_provider_when_the_agent_is_the_last_holder(
    hc, homes, holder_registry
):
    """One agent, one instance, owner unresolvable: the provider worker must be
    stopped instead of lingering until the WebUI process exits."""
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-sole")

    closed = streaming._close_evicted_agent_at_session_boundary("8613-sole", agent)

    assert closed is True
    assert provider.shutdowns == 1, "the provider worker leaked past the eviction"
    assert provider.client.closed is True


def test_a_per_instance_provider_is_released_for_every_evicted_agent(
    hc, homes, holder_registry
):
    """The documented provider contract (a new instance per agent) changes nothing:
    each agent is trivially the last holder of its own instance."""
    made: list[SharedInstanceProvider] = []
    first = _build(homes["alpha"], lambda: made.append(SharedInstanceProvider()) or made[-1],
                   "8613-fresh-a")
    second = _build(homes["beta"], lambda: made.append(SharedInstanceProvider()) or made[-1],
                    "8613-fresh-b")
    assert len(made) == 2 and made[0] is not made[1]

    for session_id, agent in (("8613-fresh-a", first), ("8613-fresh-b", second)):
        assert streaming._close_evicted_agent_at_session_boundary(session_id, agent) is True
    assert [p.shutdowns for p in made] == [1, 1]
    assert all(p.client.closed for p in made)


# ── (b) a shared instance is NOT released while another agent holds it ───────

def test_unowned_eviction_keeps_a_shared_provider_for_the_agent_still_holding_it(
    hc, homes, holder_registry
):
    """Two agents of ONE profile share the instance the plugin handed them. Evicting
    the first must not shut the instance down: the second's next memory save goes
    through the very client a teardown would close. Only once the second is alone
    may its own boundary release the provider."""
    provider = SharedInstanceProvider()
    first = _shared_agent(homes["alpha"], provider, "8613-shared-a")
    second = _shared_agent(homes["alpha"], provider, "8613-shared-b")
    assert first._memory_manager is not second._memory_manager
    assert provider.shutdowns == 0

    # Evict the first: the second agent still holds the same instance.
    assert streaming._close_evicted_agent_at_session_boundary("8613-shared-a", first) is True
    assert provider.shutdowns == 0, "a shared instance was torn down under a live agent"
    assert provider.client.closed is False
    second.save("written-after-the-co-holder-was-evicted")
    assert provider.client.writes == ["written-after-the-co-holder-was-evicted"]

    # Now the second is the last holder: its own boundary may release the provider.
    assert streaming._close_evicted_agent_at_session_boundary("8613-shared-b", second) is True
    assert provider.shutdowns == 1
    assert provider.client.closed is True


def test_the_release_is_refused_while_a_co_holder_is_alive_and_logged_once(
    hc, homes, holder_registry, caplog
):
    provider = SharedInstanceProvider()
    first = _shared_agent(homes["alpha"], provider, "8613-log-a")
    survivor = _shared_agent(homes["alpha"], provider, "8613-log-b")
    caplog.set_level(logging.WARNING)

    assert scope.agent_is_last_provider_holder(survivor)[0] is False
    assert streaming._close_evicted_agent_at_session_boundary("8613-log-a", first) is True

    shared = [r for r in caplog.records if "#8613" in r.getMessage() and "NOT released" in r.getMessage()]
    assert len(shared) == 1, [r.getMessage() for r in shared]
    assert "still held by 1 other agent" in shared[0].getMessage()
    assert provider.shutdowns == 0
    survivor.save("the-co-holders-provider-is-still-usable")
    assert provider.client.writes == ["the-co-holders-provider-is-still-usable"]


# ── the normal (owner-resolvable) path is unchanged ──────────────────────────

def test_an_owned_session_still_shuts_its_provider_down_at_the_boundary(
    hc, homes, holder_registry, caplog
):
    """A resolvable owner takes the ordinary path: ``shutdown_memory_provider``
    runs (commit + teardown) and no #8613 release is attempted."""
    provider = SharedInstanceProvider()
    session_id = "8613-owned"
    agent = _shared_agent(homes["alpha"], provider, session_id)
    lifecycle.record_owner_profile_home(session_id, str(homes["alpha"]))
    caplog.set_level(logging.INFO)

    assert streaming._close_evicted_agent_at_session_boundary(session_id, agent) is True

    assert provider.shutdowns == 1
    assert provider.client.closed is True
    assert not [r for r in caplog.records if "#8613" in r.getMessage()]


# ── the holder registry's own contract ───────────────────────────────────────

def test_construction_registers_the_agent_as_the_sole_holder(hc, homes, holder_registry):
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-reg")

    ok, why = scope.agent_is_last_provider_holder(agent)
    assert ok is True, why
    assert scope.register_provider_holders(agent, home_key=_key(homes["alpha"])) == 0, (
        "re-registering the same agent must not double-count it"
    )


def test_a_memoryless_build_registers_no_holder(hc, homes, holder_registry):
    """A build with no memory manager holds nothing, so nothing is registered and
    the agent stays cacheable exactly as before #8517."""

    class MemorylessAgent:
        def __init__(self, session_id=None, skip_memory=False, **kwargs):
            self._memory_manager = None

    agent, cacheable = scope.construct_profile_scoped_agent(
        MemorylessAgent, {"model": "m"}, profile_home=str(homes["alpha"]), session_id="8613-none"
    )
    assert cacheable is True
    assert scope._provider_instances(agent) == []
    assert scope.register_provider_holders(agent) == 0
    assert scope.agent_is_last_provider_holder(agent) == (True, "")


def test_a_collected_agent_stops_holding_the_instance(hc, homes, holder_registry):
    """An agent that dies without a boundary must not pin the instance: the
    weak-reference callback drops it, so the next boundary can prove sole
    ownership."""
    import gc  # noqa: PLC0415

    provider = SharedInstanceProvider()
    gone = [_shared_agent(homes["alpha"], provider, "8613-dead")]
    survivor = _shared_agent(homes["alpha"], provider, "8613-alive")
    assert scope.agent_is_last_provider_holder(survivor)[0] is False

    del gone[0]
    gone.clear()
    gc.collect()

    ok, why = scope.agent_is_last_provider_holder(survivor)
    assert ok is True, why
    assert not [e for e in scope._provider_holders.values() if any(ref() is None for ref in e.holders.values())]


def test_a_detached_agent_stops_holding_the_instance(hc, homes, holder_registry):
    """#8517's bound-home mismatch detaches the manager; the agent must not keep
    blocking another agent's release of the instance it was handed."""
    provider = SharedInstanceProvider()
    survivor = _shared_agent(homes["alpha"], provider, "8613-detach-survivor")

    class MismatchAgent:
        def __init__(self, session_id=None, skip_memory=False, **kwargs):
            self.session_id = session_id
            self._memory_manager = None
            manager = mm_mod.MemoryManager()
            manager.add_provider(provider)
            # Bound to beta while the construction scope holds alpha.
            manager.initialize_for_agent(
                session_id=session_id, hermes_home=str(homes["beta"]), agent_identity="beta"
            )
            self._memory_manager = manager

    agent, cacheable = scope.construct_profile_scoped_agent(
        MismatchAgent, {"model": "m"}, profile_home=str(homes["alpha"]), session_id="8613-detach"
    )
    assert cacheable is False
    assert agent._memory_manager is None
    ok, why = scope.agent_is_last_provider_holder(survivor)
    assert ok is True, why


def test_an_untracked_or_unreadable_provider_is_never_released(hc, homes, holder_registry):
    """Every unknown answers False: a provider that keeps running is the status
    quo, a provider shut down under a live agent is not."""

    class _Mgr:
        providers = "not-iterable-as-a-list-of-instances"

    class _Agent:
        def __init__(self):
            self._memory_manager = _Mgr()

    agent = _Agent()
    ok, why = scope.agent_is_last_provider_holder(agent)
    assert ok is False and "provider" in why

    class _NoManager:
        _memory_manager = None

    assert scope.agent_is_last_provider_holder(_NoManager()) == (True, "")
    assert scope.agent_is_last_provider_holder(None) == (True, "")


def test_a_release_that_cannot_be_scoped_never_runs(hc, homes, holder_registry, monkeypatch, caplog):
    """An Agent without the profile-home override cannot hold the provider's own
    profile for the drain, so nothing runs."""
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-noscope")
    monkeypatch.setattr(scope, "_enter_profile_home", lambda home, key: (scope._noop, "no override"))
    caplog.set_level(logging.WARNING)

    assert scope.release_unowned_memory_provider(agent, session_id="8613-noscope") is False
    assert provider.shutdowns == 0
    assert any("was not released" in r.getMessage() for r in caplog.records)


def test_a_manager_without_release_soft_is_left_alone(hc, homes, holder_registry, caplog):
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-old")
    agent._memory_manager.release_soft = None  # an Agent older than 2026-10-19
    caplog.set_level(logging.WARNING)

    assert scope.release_unowned_memory_provider(agent, session_id="8613-old") is False
    assert provider.shutdowns == 0
    assert any("release_soft" in r.getMessage() for r in caplog.records)


def test_an_owned_session_is_never_released_without_a_session_end(
    hc, homes, holder_registry, monkeypatch
):
    """A resolvable owner takes the ordinary path; the release is not even
    attempted."""
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-owner-known")
    monkeypatch.setattr(scope, "session_owner_profile_home", lambda sid: str(homes["alpha"]))

    assert scope.release_unowned_memory_provider(agent, session_id="8613-owner-known") is False
    assert provider.shutdowns == 0


def test_an_unreadable_binding_is_never_released(hc, homes, holder_registry, caplog):
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-unbound")
    agent._memory_manager._bound_scope = None
    caplog.set_level(logging.WARNING)

    assert scope.release_unowned_memory_provider(agent, session_id="8613-unbound") is False
    assert provider.shutdowns == 0
    assert any("does not report the profile home" in r.getMessage() for r in caplog.records)


def test_concurrent_boundaries_never_release_an_instance_another_agent_still_holds(
    hc, homes, holder_registry
):
    """Eight agents of one profile share one instance and each reaches its own
    boundary from its own thread. At most ONE release may run, and only for an
    agent that is alone — a manager-level teardown under a live agent is exactly
    what #8636 reverted."""
    provider = SharedInstanceProvider()
    agents = [_shared_agent(homes["alpha"], provider, f"8613-race-{i}") for i in range(8)]
    assert len(provider.inits) == 8
    released: list[int] = []
    lock = threading.Lock()
    barrier = threading.Barrier(len(agents))

    def boundary(i):
        agent = agents[i]
        barrier.wait(10)
        ok, why, claimed = scope._claim_last_held_teardown(agent)
        if not ok:
            scope.unregister_provider_holders(agent)
            return
        with lock:
            released.append(i)
        try:
            # The teardown the claim authorises (release_soft -> shutdown_all).
            agent._memory_manager.release_soft()
        finally:
            scope._settle_last_held_teardown(agent, claimed)

    threads = [threading.Thread(target=boundary, args=(i,)) for i in range(len(agents))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert not any(t.is_alive() for t in threads)

    assert len(released) <= 1, released
    assert provider.shutdowns == len(released)
    assert not [e for e in scope._provider_holders.values() if e.holders], "the holder table leaked"


def test_a_teardown_claim_blocks_a_second_release_of_the_same_instance(
    hc, homes, holder_registry
):
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-claim")

    ok, why, claimed = scope._claim_last_held_teardown(agent)
    assert ok is True, why
    ok_again, why_again, _ = scope._claim_last_held_teardown(agent)
    assert ok_again is False and "already running" in why_again

    scope._settle_last_held_teardown(agent, claimed)
    ok_third, why_third, _ = scope._claim_last_held_teardown(agent)
    assert ok_third is False and "not tracked" in why_third


def test_a_provider_instance_that_cannot_be_weakly_referenced_is_still_tracked(
    hc, homes, holder_registry
):
    class _SlotProvider:  # no __weakref__: cannot be weakly referenced
        __slots__ = ()
        name = "8613slot"

    provider = _SlotProvider()

    class _Mgr:
        def __init__(self):
            self.providers = [provider]
            self._bound_scope = {"hermes_home": str(homes["alpha"]), "home_key": _key(homes["alpha"])}

    class _Agent:
        def __init__(self):
            self._memory_manager = _Mgr()

    agent = _Agent()
    assert scope.register_provider_holders(agent, home_key=_key(homes["alpha"])) == 1
    ok, why = scope.agent_is_last_provider_holder(agent)
    assert ok is True, why
    # A second holder still blocks the release, and dropping this one frees it.
    other = _Agent()
    assert scope.register_provider_holders(other, home_key=_key(homes["alpha"])) == 1
    assert scope.agent_is_last_provider_holder(agent)[0] is False
    scope.unregister_provider_holders(agent)
    ok, why = scope.agent_is_last_provider_holder(other)
    assert ok is True, why


def test_a_stale_instance_id_is_never_trusted_for_a_new_instance(hc, homes, holder_registry):
    """A collected instance's ``id()`` can be reused; the entry must describe the
    instance it was recorded for, never the new one."""
    import gc  # noqa: PLC0415

    class _Prov:
        name = "8613stale"

    class _Mgr:
        def __init__(self, provider):
            self.providers = [provider]
            self._bound_scope = {"hermes_home": str(homes["alpha"]), "home_key": _key(homes["alpha"])}

    class _Agent:
        def __init__(self, provider):
            self._memory_manager = _Mgr(provider)

    first = _Prov()
    holder = _Agent(first)
    scope.register_provider_holders(holder, home_key=_key(homes["alpha"]))
    stale_key = id(first)
    # The manager is the last strong reference to the instance (as in the Agent:
    # ``MemoryManager._providers``); dropping both lets it be collected.
    holder._memory_manager = None
    del first
    gc.collect()
    assert scope._provider_holders.get(stale_key) is None, "a collected instance must leave the table"

    second = _Prov()  # may reuse the id
    other = _Agent(second)
    assert scope.register_provider_holders(other, home_key=_key(homes["alpha"])) == 1
    entry = scope._provider_holders.get(id(second))
    assert entry is not None and entry.ref() is second
    ok, why = scope.agent_is_last_provider_holder(other)
    assert ok is True, why


# ── the wiring the fix depends on ────────────────────────────────────────────

def test_the_eviction_path_imports_the_release_and_the_holder_cleanup():
    src = (scope.__file__ or "").replace("memory_profile_scope.py", "streaming.py")
    import pathlib  # noqa: PLC0415

    text = pathlib.Path(src).read_text(encoding="utf-8")
    assert "release_unowned_memory_provider(" in text
    assert "unregister_provider_holders(agent)" in text
    # The release runs only when the owner-scoped shutdown was refused.
    assert text.index("if not call_as_session_owner(") < text.index("release_unowned_memory_provider(")
