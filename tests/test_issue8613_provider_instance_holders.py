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

Acquisition and teardown are ONE instance-lifetime protocol, and these tests
replay the two windows where a construction overlaps a teardown: a holder acquired
while the release is mid-flight (schedule 1) and an eviction that lands after the
new constructor's native initialisation but before it registers (schedule 2). The
oracle in both is the SURVIVING agent's next save — a settlement warning reports
the damage after it happened, and the pre-registration window produces no warning
at all, so neither is proof on its own.
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

def make_holder_agent_cls(provider_for_build, *, home, identity="alpha", after_init=None):
    """Stand-in AIAgent. ``__init__`` binds memory the way the real Agent does
    (``MemoryManager.initialize_for_agent`` -> ``claim_instance`` -> ``initialize``);
    ``shutdown_memory_provider`` is ``run_agent.AIAgent.shutdown_memory_provider``
    (claim teardown -> ``on_session_end`` -> ``shutdown_all``); ``save`` is one
    memory write through the agent's provider.

    ``after_init`` runs at the very end of ``__init__``, AFTER the native manager
    has been initialised and BEFORE the constructor returns — the window in which
    the holder is not registered yet (``construct_profile_scoped_agent`` registers
    it only once ``agent_cls(...)`` has returned).
    """

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
                if after_init is not None:
                    after_init()
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


def _build(home, provider_for_build, session_id, *, identity="alpha", after_init=None):
    """One agent built through the production constructor, so the holder table is
    populated exactly the way a real turn populates it."""
    agent, cacheable = scope.construct_profile_scoped_agent(
        make_holder_agent_cls(
            provider_for_build, home=home, identity=identity, after_init=after_init
        ),
        {"model": "m"},
        profile_home=str(home),
        session_id=session_id,
    )
    assert cacheable is True, "a memory-bearing build must stay cacheable"
    assert agent._memory_manager is not None
    return agent


def _shared_agent(home, provider, session_id):
    return _build(home, lambda: provider, session_id)


def surviving_save(agent, marker):
    """ORACLE for a shared instance torn down under a live agent: the surviving
    agent's NEXT memory save — a memory tool call or an autosave in a later turn.

    Returns the provider the save went through, or ``None`` when the shared
    instance was refused to this agent (it then keeps built-in memory only).
    Raises the issue's own failure, ``memory provider client is closed: the save
    failed``, when the instance was shut down underneath an agent that holds it.

    The settlement warning and the release counters only report the damage after
    it happened, so neither is accepted as proof on its own: the race is only
    closed when the surviving agent's next save still works.
    """
    manager = getattr(agent, "_memory_manager", None)
    provider = manager.get_provider(PROVIDER_NAME) if manager is not None else None
    if provider is None:
        return None
    provider.remember(marker)
    return provider


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


# ── (c) the two in-flight schedules: acquisition DURING a teardown ───────────
#
# Both replay a construction that overlaps a teardown. In each one the shared
# provider instance ends up shut down under an agent that is still live, and the
# only signal is the surviving agent's NEXT save — the settlement warning fires
# after the damage and the pre-registration window produces no warning at all.

def test_a_holder_acquired_during_the_release_is_not_shut_down_with_it(
    hc, homes, holder_registry, monkeypatch
):
    """Schedule 1 — insert a new holder while the release is running.

    The old agent's soft release is paused at its REAL native flush
    (``MemoryManager.flush_pending``), with the teardown claim already held; a new
    same-profile agent is then constructed and registered; the release resumes.
    The instance must not be handed to the new agent — a client the release closes
    is a save that fails for the agent that just acquired it.
    """
    provider = SharedInstanceProvider()
    old = _shared_agent(homes["alpha"], provider, "8613-s1-old")
    assert scope.agent_is_last_provider_holder(old)[0] is True

    at_flush = threading.Event()
    resume = threading.Event()
    real_flush = mm_mod.MemoryManager.flush_pending

    def gated_flush(self, timeout=None):
        # The real native flush; only the evicted agent's manager is paused.
        if self is old._memory_manager:
            at_flush.set()
            assert resume.wait(30), "the paused release was never resumed"
        return real_flush(self, timeout)

    monkeypatch.setattr(mm_mod.MemoryManager, "flush_pending", gated_flush)

    closed: list[bool] = []

    def evict():
        closed.append(streaming._close_evicted_agent_at_session_boundary("8613-s1-old", old))

    thread = threading.Thread(target=evict, name="8613-s1-eviction", daemon=True)
    thread.start()
    assert at_flush.wait(30), "the release never reached its native flush"
    assert provider.shutdowns == 0, "the provider was shut down before the release finished"

    # Acquire the shared instance while the old agent's release is mid-flight.
    survivor = _shared_agent(homes["alpha"], provider, "8613-s1-new")

    resume.set()
    thread.join(30)
    assert not thread.is_alive()
    assert closed == [True]

    # ORACLE: the surviving agent's next save must not fail.
    used = surviving_save(survivor, "8613-s1-save-after-the-release")
    if used is None:
        assert provider.client.closed is True, "the release did not run"
    else:
        assert used.client.closed is False, "the provider was shut down under a live holder"
    entry = scope._provider_holders.get(id(provider))
    assert entry is None or id(survivor) not in entry.holders, (
        "a holder was recorded for an instance that was torn down"
    )


def test_an_eviction_during_construction_never_tears_the_instance_down(
    hc, homes, holder_registry
):
    """Schedule 2 — insert an eviction before the new holder is registered.

    The new constructor has finished its native initialisation (the instance is
    claimed and initialised) but has not returned, so the holder is not in the
    table yet. Evicting the old sole RECORDED holder here must not tear the
    instance down: the new agent is returned cacheable and its next save goes
    through the very client a teardown would close. This window produces no
    gained-holder warning at all, so the save is the only oracle.
    """
    provider = SharedInstanceProvider()
    old = _shared_agent(homes["alpha"], provider, "8613-s2-old")

    initialised = threading.Event()
    proceed = threading.Event()

    def pause_after_native_init():
        initialised.set()
        assert proceed.wait(30), "the paused construction was never resumed"

    built: list = []

    def build():
        built.append(
            _build(
                homes["alpha"],
                lambda: provider,
                "8613-s2-new",
                after_init=pause_after_native_init,
            )
        )

    thread = threading.Thread(target=build, name="8613-s2-construction", daemon=True)
    thread.start()
    assert initialised.wait(30), "the constructor never initialised its native manager"
    assert provider.shutdowns == 0

    # The old agent is the sole RECORDED holder — the snapshot the proof would have
    # used before the reservation existed. Capture what the full proof answers here,
    # while the construction is paused: it must refuse, because a construction for
    # the same profile home is acquiring the instance right now.
    entry = scope._provider_holders.get(id(provider))
    recorded_solely = entry is not None and list(entry.holders) == [id(old)]
    ok, why = scope.agent_is_last_provider_holder(old)
    proof_refused = ok is False and "in flight" in why

    assert streaming._close_evicted_agent_at_session_boundary("8613-s2-old", old) is True

    proceed.set()
    thread.join(30)
    assert not thread.is_alive()
    survivor = built[0]

    # ORACLE: the surviving agent's next save must not fail.
    used = surviving_save(survivor, "8613-s2-save-after-the-eviction")
    assert used is not None, "the surviving agent lost the provider it had acquired"
    assert used.client.closed is False
    assert provider.client.writes == ["8613-s2-save-after-the-eviction"]

    # The proof that made the oracle pass, and the release that therefore never ran.
    assert recorded_solely, "the recorded snapshot was not what this schedule assumes"
    assert proof_refused, why
    assert provider.shutdowns == 0, "the provider was torn down under the agent being built"


# ── (d) the rest of the instance-lifetime protocol ──────────────────────────

def test_a_build_that_lands_on_a_tearing_down_instance_fails_closed(
    hc, homes, holder_registry
):
    """An Agent without the ``claim_instance`` seam keeps the instance its manager
    was handed, so the reservation alone cannot stop it binding a dying instance.
    The construction must then fail closed: the agent is rebuilt without memory
    and not cached, so the next turn constructs it again once the teardown settled.
    """
    provider = SharedInstanceProvider()
    first = _shared_agent(homes["alpha"], provider, "8613-failclosed-a")
    ok, why, claimed = scope._claim_last_held_teardown(first)
    assert ok is True, why  # a teardown is now running; nothing has been released yet

    class SeamlessAgent:
        """An Agent older than 2026-10-17: it binds the provider without ever
        calling ``claim_instance``, so the P4 acquisition guard never sees it."""

        def __init__(self, session_id=None, skip_memory=False, **kwargs):
            self.session_id = session_id
            self._memory_manager = None
            if not skip_memory:
                manager = mm_mod.MemoryManager()
                manager._providers = [provider]
                manager._bound_scope = {
                    "hermes_home": str(homes["alpha"]),
                    "home_key": _key(homes["alpha"]),
                }
                self._memory_manager = manager

    agent, cacheable = scope.construct_profile_scoped_agent(
        SeamlessAgent, {"model": "m"}, profile_home=str(homes["alpha"]), session_id="8613-failclosed-b"
    )
    assert cacheable is False, "an agent holding a dying instance must not be cached"
    assert agent._memory_manager is None, "the dying instance was left on the agent"
    assert provider.shutdowns == 0
    assert scope.register_provider_holders(agent) == 0

    # Once the teardown settles the instance is live again and may be acquired.
    scope._settle_last_held_teardown(first, claimed)
    again = _shared_agent(homes["alpha"], provider, "8613-failclosed-c")
    assert again._memory_manager.get_provider(PROVIDER_NAME) is provider
    assert scope.agent_is_last_provider_holder(again)[0] is True


def test_a_torn_down_instance_is_never_acquired_again(hc, homes, holder_registry):
    """The release shut the shared instance down, so a LATER construction for the
    same profile must not be handed the client it just closed — the surviving
    agent's next save would otherwise fail with no release anywhere near it."""
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-retire")

    assert streaming._close_evicted_agent_at_session_boundary("8613-retire", agent) is True
    assert provider.shutdowns == 1 and provider.client.closed is True

    later = _shared_agent(homes["alpha"], provider, "8613-retire-later")
    assert later._memory_manager.get_provider(PROVIDER_NAME) is None, (
        "a torn-down instance was handed to a new agent"
    )
    assert scope.register_provider_holders(later, home_key=_key(homes["alpha"])) == 0
    # The retired entry survives the eviction's holder cleanup, so the refusal
    # keeps working for every later construction.
    entry = scope._provider_holders.get(id(provider))
    assert entry is not None and entry.retired is True and not entry.holders
    assert scope.agent_is_last_provider_holder(later) == (True, "")


def test_an_aborted_construction_releases_its_reservation(hc, homes, holder_registry):
    """A build that raises after the reservation was taken must not fence this
    profile's releases for the rest of the process."""

    class Boom:
        def __init__(self, **kwargs):
            raise RuntimeError("construction failed")

    with pytest.raises(RuntimeError, match="construction failed"):
        scope.construct_profile_scoped_agent(
            Boom, {"model": "m"}, profile_home=str(homes["alpha"]), session_id="8613-abort"
        )
    assert scope._inflight_homes == {}, scope._inflight_homes


def test_a_construction_for_another_profile_does_not_fence_this_release(
    hc, homes, holder_registry
):
    """The reservation is per profile home: a construction for beta must neither
    stop alpha's last holder from releasing nor wedge it behind beta's build."""
    provider = SharedInstanceProvider()
    agent = _shared_agent(homes["alpha"], provider, "8613-xprofile")

    started = threading.Event()
    proceed = threading.Event()

    def pause_after_native_init():
        started.set()
        assert proceed.wait(30), "the paused construction was never resumed"

    built: list = []

    def build_beta():
        built.append(
            _build(
                homes["beta"],
                SharedInstanceProvider,
                "8613-xprofile-beta",
                after_init=pause_after_native_init,
            )
        )

    thread = threading.Thread(target=build_beta, name="8613-xprofile-build", daemon=True)
    thread.start()
    assert started.wait(30), "the other profile's construction never started"

    assert scope.agent_is_last_provider_holder(agent)[0] is True, (
        "another profile's construction fenced this profile's release"
    )
    assert scope.release_unowned_memory_provider(agent, session_id="8613-xprofile") is True
    assert provider.shutdowns == 1 and provider.client.closed is True

    proceed.set()
    thread.join(30)
    assert not thread.is_alive()
    assert built[0]._memory_manager.get_provider(PROVIDER_NAME) is not None


def test_the_reservation_counts_concurrent_constructions_of_one_profile(
    hc, homes, holder_registry
):
    """Two constructions for the same profile home hold the fence together and
    release it only when the last of them finishes."""
    key = _key(homes["alpha"])
    scope._reserve_acquisition(key)
    scope._reserve_acquisition(key)
    assert scope._inflight_acquisitions(key) == 2
    scope._end_acquisition(key)
    assert scope._inflight_acquisitions(key) == 1
    scope._end_acquisition(key)
    assert scope._inflight_acquisitions(key) == 0
    assert scope._inflight_homes == {}
    # An unkeyed lookup cannot be correlated, so any construction in flight counts.
    scope._reserve_acquisition(key)
    assert scope._inflight_acquisitions("") == 1
    scope._end_acquisition(key)


def test_an_unkeyed_instance_is_fenced_by_any_construction_in_flight(
    hc, homes, holder_registry
):
    """An instance whose profile home cannot be correlated must not be released
    while ANY construction is in flight: an unprovable answer is not a proof."""

    class _Prov:
        name = "8613unkeyed"

    class _Mgr:
        def __init__(self):
            self.providers = [_Prov()]
            self._bound_scope = {"hermes_home": str(homes["alpha"])}
            self.release_soft = lambda: None
            self.claim_teardown = lambda *, session_end: True

    class _Agent:
        def __init__(self):
            self._memory_manager = _Mgr()

    agent = _Agent()
    scope.register_provider_holders(agent, home_key="")
    assert scope.agent_is_last_provider_holder(agent)[0] is True

    scope._reserve_acquisition(_key(homes["beta"]))
    try:
        ok, why = scope.agent_is_last_provider_holder(agent)
        assert ok is False and "in flight" in why, why
    finally:
        scope._end_acquisition(_key(homes["beta"]))
    assert scope.agent_is_last_provider_holder(agent)[0] is True


# ── the wiring the fix depends on ────────────────────────────────────────────

def test_the_eviction_path_imports_the_release_and_the_holder_cleanup():
    src = (scope.__file__ or "").replace("memory_profile_scope.py", "streaming.py")
    import pathlib  # noqa: PLC0415

    text = pathlib.Path(src).read_text(encoding="utf-8")
    assert "release_unowned_memory_provider(" in text
    assert "unregister_provider_holders(agent)" in text
    # The release runs only when the owner-scoped shutdown was refused.
    assert text.index("if not call_as_session_owner(") < text.index("release_unowned_memory_provider(")
