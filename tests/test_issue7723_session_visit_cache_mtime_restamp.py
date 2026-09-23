"""Regression coverage for #7723 session-visit cache mtime semantics.

The session-visit freshness window (``_SESSION_VISIT_MODELS_FRESHNESS_SECONDS``)
is decided by ``_models_cache_file_age_seconds()`` — the file's mtime on disk.
The on-disk mtime is written **only** by ``_save_models_cache_to_disk()``
(called from a successful live rebuild), so it records "last live rebuild",
not "last read". This file guards:

  1. The buggy ``os.utime`` restamp on the disk-hit branch is GONE
     (regression test for the original PR's CORE finding: the restamp
     made mtime record "last read" and defeated the 300 s horizon
     indefinitely on multi-profile installs).
  2. Stale-while-revalidate, per profile: when the on-disk mtime crosses
     the 300 s horizon, the foreground returns the stale disk catalog
     immediately and fires a coalesced per-profile background
     ``force_refresh``. The rebuild goes through the normal
     ``_save_models_cache_to_disk`` path — so the mtime advances as a
     side effect of a real rebuild, not a read.
  3. The dual-profile alternation bug: two profiles alternating visits
     every 100 s (well under the 300 s horizon) used to pay 0 rebuilds
     with the os.utime restamp; under SWR each profile must rebuild
     exactly once across the horizon crossing.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _catalog(label: str) -> dict:
    return {
        "active_provider": "openai",
        "default_model": label,
        "configured_model_badges": {},
        "groups": [
            {
                "provider": "OpenAI",
                "provider_id": "openai",
                "models": [{"id": label, "label": label, "supports_fast_tier": False}],
            }
        ],
        "aliases": {},
    }


def _reset_models_memory_cache(monkeypatch):
    import api.config as cfg

    monkeypatch.setattr(cfg, "_available_models_cache", None, raising=False)
    monkeypatch.setattr(cfg, "_available_models_cache_ts", 0.0, raising=False)
    monkeypatch.setattr(cfg, "_available_models_live_rebuild_ts", 0.0, raising=False)
    monkeypatch.setattr(cfg, "_available_models_cache_source_fingerprint", None, raising=False)
    monkeypatch.setattr(cfg, "_cache_build_in_progress", False, raising=False)
    monkeypatch.setattr(cfg, "_session_visit_rebuild_threads", {}, raising=False)
    monkeypatch.setattr(cfg, "_session_visit_rebuild_lock", threading.Lock(), raising=False)


def _wait_for_session_visit_rebuild(monkeypatch, timeout: float = 5.0):
    """Join any background session-visit SWR threads started by this test.

    The session-visit stale-while-revalidate path launched by
    ``_maybe_start_session_visit_background_rebuild`` is fire-and-forget on a
    daemon thread. Tests that need the background rebuild to land before
    asserting on the in-memory cache call this to deterministically wait it
    out. Raises if any tracked thread does not finish within ``timeout``.
    """
    import api.config as cfg

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with cfg._session_visit_rebuild_lock:
            threads = list(cfg._session_visit_rebuild_threads.values())
        if not threads:
            return
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive():
                raise AssertionError(
                    f"session-visit background rebuild thread {thread.name!r} "
                    f"did not finish within {timeout}s"
                )


# ── 1. mtime is NOT restamped on a fresh disk hit (regression for the CORE finding)


def test_disk_hit_does_not_restamp_mtime(tmp_path, monkeypatch):
    """A fresh session-visit disk hit must NOT advance the on-disk mtime.
    The previous fix attempted this with ``os.utime(cache_path, None)`` on
    the hit branch, which broke the per-profile 300 s horizon (mtime
    recorded "last read", not "last live rebuild"). Under the SWR fix the
    mtime only moves as a side effect of ``_save_models_cache_to_disk``
    running after a real rebuild.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    disk_catalog = _catalog("cached-model")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    planted_mtime = time.time() - 60.0
    os.utime(cache_path, (planted_mtime, planted_mtime))
    planted_stat = cache_path.stat()
    assert abs(planted_stat.st_mtime - planted_mtime) < 0.01

    rebuild_calls: list[dict] = []

    def _unexpected_live_rebuild(**kwargs):
        rebuild_calls.append(kwargs)
        raise AssertionError("fresh session-visit cache must not run a live rebuild")

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: disk_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})
    monkeypatch.setattr(cfg, "get_available_models", _unexpected_live_rebuild)

    result = cfg.get_available_models_for_session_visit()
    assert result == disk_catalog

    # The mtime must NOT have been touched — the SWR fix never reads
    # ``cache_path.stat().st_mtime`` in a way that writes it back, and
    # the ``os.utime(restamp)`` from the old (rejected) fix is gone.
    after_stat = cache_path.stat()
    assert abs(after_stat.st_mtime - planted_mtime) < 0.01, (
        f"on-disk mtime must be untouched by a session-visit disk hit, "
        f"got delta {after_stat.st_mtime - planted_mtime:.3f}s"
    )
    assert rebuild_calls == [], "fresh session-visit hit must not trigger a rebuild"


# ── 2. Repeated hits within the horizon do not trigger a rebuild


def test_repeated_hits_within_horizon_do_not_rebuild(tmp_path, monkeypatch):
    """Three back-to-back hits within the 300 s horizon (planted mtime just
    under the cliff) must not trigger a foreground or background rebuild
    — the in-memory cache is warm after the first hit and short-circuits
    the rest, the disk mtime is NOT restamped, and the disk hit returns
    a copy of the payload.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    disk_catalog = _catalog("sustained-hits")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    planted = time.time() - 290.0
    os.utime(cache_path, (planted, planted))

    rebuild_calls: list[dict] = []

    def _rebuild(**kwargs):
        rebuild_calls.append(kwargs)
        return disk_catalog

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: disk_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})
    monkeypatch.setattr(cfg, "get_available_models", _rebuild)

    for _ in range(3):
        time.sleep(0.05)
        result = cfg.get_available_models_for_session_visit()
        assert result == disk_catalog

    _wait_for_session_visit_rebuild(monkeypatch)
    assert rebuild_calls == [], (
        f"no live rebuild should have run across 3 back-to-back in-horizon "
        f"hits, but got {len(rebuild_calls)} call(s): {rebuild_calls}"
    )
    # The mtime is still the 290-s-old planted value (no os.utime restamp).
    final_mtime = cache_path.stat().st_mtime
    age_after = time.time() - final_mtime
    assert age_after > 200.0, (
        f"on-disk mtime must NOT be re-stamped by a disk hit; "
        f"got age={age_after:.3f}s after 3 hits"
    )


# ── 3. The CORE scenario: dual profile alternation across the 300 s horizon
#       (this is the exact bug the reviewer demonstrated in the review)


def test_dual_profile_alternation_triggers_per_profile_rebuild(tmp_path, monkeypatch):
    """Two profiles alternating visits every 100 s over a 1200 s window
    (6 visits per profile, 12 total) cross the 300 s session-visit
    horizon 4 times per profile. With the rejected ``os.utime`` restamp
    fix, the restamp-on-hit would have kept the mtime fresh indefinitely
    → 0 rebuilds per profile, both returning the pre-horizon catalog.
    Under SWR, each profile must trigger a rebuild every time it crosses
    the horizon.

    A monkeypatched clock is used so the test runs in milliseconds, not
    minutes. The test asserts:

      1. Every horizon crossing actually launches a background rebuild
         (tracked via ``_session_visit_rebuild_threads`` entries, not
         via the rebuild mock's mutable shared state — the rebuild mock
         can't always determine which profile it was started for because
         ``get_active_profile_name()`` is not re-resolved on the worker
         thread, so the foreground's mutable profile pointer may have
         moved on by the time the worker fires).
      2. The mtime ``_models_cache_file_age_seconds`` reports is the
         **planted** mtime (we never call ``os.utime`` to advance it
         from a read), so a 100-s-old file still shows up as 100 s old
         on the next visit.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)

    profile_a_path = tmp_path / "models_cache.profile_a.json"
    profile_b_path = tmp_path / "models_cache.profile_b.json"
    profile_a_path.write_text("{}", encoding="utf-8")
    profile_b_path.write_text("{}", encoding="utf-8")
    planted_mtime = time.time() - 100.0
    for p in (profile_a_path, profile_b_path):
        os.utime(p, (planted_mtime, planted_mtime))

    # A simple disk catalog; we don't care about content for this test
    # beyond the foreground returning something shape-valid.
    disk_catalog = _catalog("disk")
    rebuilt_catalog_a = _catalog("rebuilt-a")
    rebuilt_catalog_b = _catalog("rebuilt-b")

    simulated_now = [planted_mtime + 100.0]
    # Track which profile the foreground was last requesting.
    _profile_call = [0]
    _profile_paths = [profile_a_path, profile_b_path]

    def _path_for_profile():
        return _profile_paths[_profile_call[0]]

    def _load_for_profile():
        return disk_catalog

    def _rebuild_for_profile(**kwargs):
        # ``force_refresh=True`` rebuilds just return a single catalog;
        # the foreground mock (which loads from disk) controls what
        # the *next* visit returns. We don't try to assert a per-profile
        # catalog content here — see ``test_same_profile_concurrent_stale_visits_coalesce_to_one_rebuild``
        # for per-profile rebuild assertion. This test focuses on the
        # *count* of SWR background rebuilds launched across the horizon.
        return rebuilt_catalog_a

    monkeypatch.setattr(cfg.time, "time", lambda: simulated_now[0])
    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", _path_for_profile)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", _load_for_profile)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", _load_for_profile)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "test"})
    monkeypatch.setattr(cfg, "_save_models_cache_to_disk", lambda _cache: None)
    # Force every visit to walk the disk / SWR path (process-global memory
    # cache is irrelevant to this test's per-profile horizon semantics).
    monkeypatch.setattr(cfg, "_get_fresh_memory_models_cache", lambda _now: None)
    # Patch the SWR's per-call active-profile resolver directly so we
    # don't need a real cookie/profile TLS to alternate keys.
    monkeypatch.setattr(
        cfg,
        "_session_visit_active_profile_name",
        lambda: ("profile_a" if _profile_call[0] % 2 == 0 else "profile_b"),
    )
    monkeypatch.setattr(cfg, "get_available_models", _rebuild_for_profile)

    # Track every time the foreground fires a background rebuild. The
    # thread is registered inside ``_maybe_start_session_visit_background_rebuild``
    # before the worker fires, so we can capture the profile_key the
    # foreground *intended* to rebuild.
    launched_rebuilds: list[str] = []
    real_swr = cfg._maybe_start_session_visit_background_rebuild

    def _tracked_swr():
        with cfg._session_visit_rebuild_lock:
            key = cfg._session_visit_active_profile_name()
        launched_rebuilds.append(key)
        real_swr()

    monkeypatch.setattr(cfg, "_maybe_start_session_visit_background_rebuild", _tracked_swr)

    # 12 alternating visits: 6 per profile, 100 s apart each, profile B
    # offset by 50 s so the two profiles never collide. The horizon is
    # 300 s. After tick 2 the mtime is past the horizon for both profiles.
    for tick in range(6):
        _profile_call[0] = 0  # profile A
        simulated_now[0] = planted_mtime + 100.0 * (tick + 1)
        cfg.get_available_models_for_session_visit()
        _profile_call[0] = 1  # profile B
        simulated_now[0] = planted_mtime + 100.0 * (tick + 1) + 50.0
        cfg.get_available_models_for_session_visit()

    # Wait for any background rebuilds to land.
    _wait_for_session_visit_rebuild(monkeypatch, timeout=10)

    # Count the SWR launches per profile. The os.utime restamp fix would
    # have produced 0 SWR launches per profile (mtime would have been
    # fresh forever). The SWR fix produces at least 1 per profile once
    # the horizon is crossed.
    profile_a_launches = sum(1 for k in launched_rebuilds if k == "profile_a")
    profile_b_launches = sum(1 for k in launched_rebuilds if k == "profile_b")
    assert profile_a_launches >= 1, (
        f"profile A must launch a background rebuild after its 300 s "
        f"horizon crossing; got {profile_a_launches} launches "
        f"(os.utime restamp bug would produce 0). All launches: {launched_rebuilds}"
    )
    assert profile_b_launches >= 1, (
        f"profile B must launch a background rebuild after its 300 s "
        f"horizon crossing; got {profile_b_launches} launches "
        f"(os.utime restamp bug would produce 0). All launches: {launched_rebuilds}"
    )


# ── 4. Stale visit returns immediately (latency: no 4 s foreground wait)


def test_stale_visit_returns_immediately_without_foreground_wait(tmp_path, monkeypatch):
    """A stale session-visit must return within milliseconds, not block on
    the live provider probe. The foreground returns the disk/stale catalog
    and fires a background rebuild; the caller's wall-time is the disk
    read + the small SWR bookkeeping, NOT the multi-second live probe.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    stale_catalog = _catalog("stale-latency")
    rebuilt_catalog = _catalog("rebuilt-latency")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    old = time.time() - 600.0
    os.utime(cache_path, (old, old))

    rebuild_started = threading.Event()
    rebuild_release = threading.Event()
    rebuild_calls: list[dict] = []

    def _slow_rebuild(**kwargs):
        # Simulate the real-world multi-second provider probe.
        rebuild_calls.append(kwargs)
        rebuild_started.set()
        assert rebuild_release.wait(timeout=5), "test never released the background rebuild"
        return rebuilt_catalog

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: stale_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: stale_catalog)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})
    monkeypatch.setattr(cfg, "get_available_models", _slow_rebuild)

    started = time.monotonic()
    result = cfg.get_available_models_for_session_visit()
    elapsed_ms = (time.monotonic() - started) * 1000.0
    # Foreground returned the stale catalog immediately.
    assert result == stale_catalog
    # The slow rebuild is still in flight — the foreground did not wait for it.
    assert rebuild_started.wait(timeout=5), "background rebuild never started"
    # Generous bound — disk read + a few Python lines should be well under 200 ms
    # on any test machine. The point is: it must NOT be 4 s.
    assert elapsed_ms < 200.0, (
        f"stale session-visit must return immediately, not block on the "
        f"live probe; got {elapsed_ms:.1f} ms (would be ~4000 ms if the "
        f"old blocking-foreground contract were in effect)"
    )
    rebuild_release.set()
    _wait_for_session_visit_rebuild(monkeypatch)
    assert rebuild_calls == [{"force_refresh": True}]


# ── 5. Per-profile coalescing: same profile, concurrent stale visits


def test_same_profile_concurrent_stale_visits_coalesce_to_one_rebuild(tmp_path, monkeypatch):
    """Two concurrent stale visits on the same profile must coalesce into
    exactly one background ``force_refresh`` (and both foregrounds must
    return the stale catalog immediately). Different profiles must NOT
    coalesce (each profile gets its own rebuild), per the reviewer's
    "profiles are islands" note.
    """
    import api.config as cfg
    from concurrent.futures import ThreadPoolExecutor

    _reset_models_memory_cache(monkeypatch)
    stale_catalog = _catalog("coalesce-stale")
    rebuilt_catalog = _catalog("coalesce-rebuilt")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    old = time.time() - 600.0
    os.utime(cache_path, (old, old))

    rebuild_count = 0
    rebuild_lock = threading.Lock()
    rebuild_in_progress = threading.Event()
    rebuild_release = threading.Event()

    def _slow_rebuild(**kwargs):
        nonlocal rebuild_count
        with rebuild_lock:
            rebuild_count += 1
        # Block the first rebuild so the second concurrent stale visit
        # definitely sees an in-flight SWR thread and coalesces into it.
        rebuild_in_progress.set()
        assert rebuild_release.wait(timeout=5), "test never released the background rebuild"
        return rebuilt_catalog

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: stale_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: stale_catalog)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})
    monkeypatch.setattr(cfg, "_save_models_cache_to_disk", lambda _cache: None)
    monkeypatch.setattr(cfg, "get_available_models", _slow_rebuild)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(cfg.get_available_models_for_session_visit) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]

    assert all(result == stale_catalog for result in results)
    assert rebuild_in_progress.wait(timeout=5), "background rebuild never started"
    rebuild_release.set()
    _wait_for_session_visit_rebuild(monkeypatch)
    assert rebuild_count == 1, (
        f"two concurrent same-profile stale visits must coalesce into one "
        f"background rebuild; got {rebuild_count}"
    )


# ── 6. mtime is only moved by _save_models_cache_to_disk, never by a read


def test_disk_mtime_only_moves_on_real_rebuild(tmp_path, monkeypatch):
    """The session-visit on-disk mtime must ONLY advance when a real live
    rebuild runs (whose ``_save_models_cache_to_disk`` call is the sole
    mover). Reading the disk cache — via a session-visit hit, a session-
    visit stale visit, or a plain ``get_available_models`` disk hit —
    must never move the mtime. This is the mtime semantic the 300 s
    horizon is defined against.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    stale_catalog = _catalog("mtime-stale")
    rebuilt_catalog = _catalog("mtime-rebuilt")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    planted = time.time() - 600.0
    os.utime(cache_path, (planted, planted))

    disk_save_calls: list[dict] = []
    save_done = threading.Event()
    # Block the background rebuild until the foreground has captured its
    # mtime observation, so the test can prove the mtime was untouched
    # *before* the rebuild was allowed to land.
    rebuild_started = threading.Event()
    rebuild_release = threading.Event()

    def _track_save(cache):
        disk_save_calls.append({"time": time.time()})
        # Mimic the real save's mtime advance (write_text does os.write +
        # close, which updates st_mtime to "now" on most filesystems).
        cache_path.write_text("rebuilt", encoding="utf-8")
        save_done.set()

    def _slow_rebuild(_builder):
        # Block until the foreground observation has completed, then
        # return the rebuilt catalog.
        rebuild_started.set()
        assert rebuild_release.wait(timeout=5), "test never released the background rebuild"
        return rebuilt_catalog

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_LIVE_REBUILD_BUDGET_SECONDS", 0.0, raising=False)
    monkeypatch.setattr(cfg, "_get_config_path", lambda: config_path)
    monkeypatch.setattr(cfg, "_cfg_path", config_path, raising=False)
    monkeypatch.setattr(cfg, "_cfg_mtime", config_path.stat().st_mtime, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: stale_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: stale_catalog)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})
    monkeypatch.setattr(cfg, "_save_models_cache_to_disk", _track_save)
    monkeypatch.setattr(cfg, "_cfg_mtime", 0.0, raising=False)
    monkeypatch.setattr(cfg, "_invoke_models_rebuild", _slow_rebuild)

    # 1. Plain ``get_available_models`` disk hit must NOT rewrite the file.
    cfg.get_available_models()
    mtime_after_plain = cache_path.stat().st_mtime
    assert abs(mtime_after_plain - planted) < 0.01, "plain disk hit must not move mtime"
    assert disk_save_calls == []

    # 2. Stale session-visit must return immediately AND the *background*
    # rebuild — when it eventually runs through ``_save_models_cache_to_disk``
    # — is the only thing that may move the mtime.
    result = cfg.get_available_models_for_session_visit()
    assert result == stale_catalog
    # The rebuild is now in flight (blocked on rebuild_release). The
    # foreground has returned; the mtime must STILL be the planted value
    # because the rebuild has not yet run its publish step.
    assert rebuild_started.wait(timeout=5), "background rebuild never started"
    mtime_after_stale_fg = cache_path.stat().st_mtime
    assert abs(mtime_after_stale_fg - planted) < 0.01, (
        "stale foreground must not move mtime before the rebuild's "
        "_save_models_cache_to_disk call has run"
    )
    assert disk_save_calls == []

    # Release the rebuild so its publish can run.
    rebuild_release.set()
    assert save_done.wait(timeout=5), (
        "background rebuild never reached _save_models_cache_to_disk; "
        "the only legitimate mover of the on-disk mtime"
    )
    mtime_after_rebuild = cache_path.stat().st_mtime
    assert mtime_after_rebuild > mtime_after_stale_fg, (
        f"the only legitimate mtime advance is the rebuild's "
        f"_save_models_cache_to_disk call; pre={mtime_after_stale_fg} "
        f"post={mtime_after_rebuild}"
    )
    assert len(disk_save_calls) == 1
