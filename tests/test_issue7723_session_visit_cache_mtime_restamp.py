"""Regression coverage for #7723 session-visit cache mtime restamp.

The session-visit freshness window (``_SESSION_VISIT_MODELS_FRESHNESS_SECONDS``)
is decided by ``_models_cache_file_age_seconds()`` — the file's mtime on disk.
The hit branch of ``get_available_models_for_session_visit()`` previously
returned a deep-copied payload without ever touching the file, so once the
file sat idle for longer than the freshness window every subsequent session
visit was judged stale and a full live rebuild was triggered (≈4 s). The fix
calls ``os.utime(cache_path, None)`` on the disk-hit branch so the window
slides forward on every hit instead of acting as a one-shot TTL.
"""

from __future__ import annotations

import os
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


# ── 1. test_hit_restamps_disk_mtime_to_keep_freshness_window_sliding ──────


def test_hit_restamps_disk_mtime_to_keep_freshness_window_sliding(tmp_path, monkeypatch):
    """#7723 core fix: a fresh disk cache whose mtime is just under the
    session-visit TTL must be returned AND have its mtime advanced to ~now,
    so the next session visit does not pay the ≈4 s live-rebuild cost.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    disk_catalog = _catalog("cached-model")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    # Plant mtime 60 s in the past — comfortably under the 300 s window.
    planted_mtime = time.time() - 60.0
    os.utime(cache_path, (planted_mtime, planted_mtime))
    planted_stat = cache_path.stat()
    assert abs(planted_stat.st_mtime - planted_mtime) < 0.01

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: disk_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})

    def _unexpected_live_rebuild(**_kwargs):
        raise AssertionError("fresh session-visit cache must not run a live rebuild")

    monkeypatch.setattr(cfg, "get_available_models", _unexpected_live_rebuild)

    result = cfg.get_available_models_for_session_visit()
    assert result == disk_catalog

    # The mtime must have been bumped to ~now — the whole point of the fix.
    after_stat = cache_path.stat()
    age_after = time.time() - after_stat.st_mtime
    assert age_after < 5.0, (
        f"expected mtime to be re-stamped to ~now on hit, got age={age_after:.3f}s"
        f" (planted mtime {planted_mtime:.3f}, now {time.time():.3f})"
    )


# ── 2. test_hit_returns_payload_unchanged_only_mtime_advances ──────────────


def test_hit_returns_payload_unchanged_only_mtime_advances(tmp_path, monkeypatch):
    """The fix must not modify the returned catalog content — only the on-disk
    mtime may change. This guards against an accidental in-place mutation
    leaking the source-payload identity into the in-memory cache.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    disk_catalog = _catalog("payload-intact")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    old = time.time() - 120.0
    os.utime(cache_path, (old, old))

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: disk_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})

    def _unexpected_live_rebuild(**_kwargs):
        raise AssertionError("fresh session-visit cache must not run a live rebuild")

    monkeypatch.setattr(cfg, "get_available_models", _unexpected_live_rebuild)

    result = cfg.get_available_models_for_session_visit()
    # Returned dict equals the disk payload
    assert result == disk_catalog
    # Returned dict is a *copy* (mutating it must not pollute the source)
    result["default_model"] = "MUTATED"
    assert disk_catalog["default_model"] == "payload-intact"


# ── 3. test_utime_failure_does_not_break_response_path ────────────────────


def test_utime_failure_does_not_break_response_path(tmp_path, monkeypatch):
    """If ``os.utime`` raises (e.g. the file was pruned between the
    ``_load_models_cache_from_disk`` call and the re-stamp), the hit branch
    must still return the cached payload — a read-only / flaky filesystem
    must never 500 the session-visit path.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    disk_catalog = _catalog("utime-fails")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    fresh = time.time() - 30.0
    os.utime(cache_path, (fresh, fresh))

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: disk_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})

    def _exploding_utime(_path, _times):
        raise OSError("simulated EACCES on re-stamp")

    # Only the call from the hit branch — not from test setup above.
    real_utime = os.utime
    call_count = {"n": 0}

    def _flaky_utime(path, times=None, *args, **kwargs):
        call_count["n"] += 1
        # First call was the test setup (plant fresh mtime). All subsequent
        # calls come from the hit branch and must be made non-throwing for
        # the fix's try/except wrapper, but we still want them to "succeed"
        # in the sense of advancing mtime — except we model the failure here.
        if call_count["n"] > 1:
            raise OSError("simulated EACCES on re-stamp")
        return real_utime(path, times, *args, **kwargs)

    monkeypatch.setattr("os.utime", _flaky_utime, raising=False)
    # Also patch the module reference the function reaches, since ``os`` is
    # imported as a module-level name inside api/config.py.
    monkeypatch.setattr(cfg.os, "utime", _flaky_utime)

    def _unexpected_live_rebuild(**_kwargs):
        raise AssertionError("fresh session-visit cache must not run a live rebuild")

    monkeypatch.setattr(cfg, "get_available_models", _unexpected_live_rebuild)

    # Must not raise — the try/except in the fix must swallow the OSError.
    result = cfg.get_available_models_for_session_visit()
    assert result == disk_catalog


# ── 4. test_repeated_hits_keep_window_sliding_no_rebuild_needed ────────────


def test_repeated_hits_keep_window_sliding_no_rebuild_needed(tmp_path, monkeypatch):
    """Three back-to-back hits must each bump mtime; ``get_available_models``
    (the live-rebuild path) must never be invoked, because every hit stayed
    inside the freshness window thanks to the re-stamp.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    disk_catalog = _catalog("sustained-hits")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    # Plant mtime 290 s in the past — close to the 300 s cliff; without the
    # fix, the second hit would tip over and force a rebuild.
    planted = time.time() - 290.0
    os.utime(cache_path, (planted, planted))

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: disk_catalog)
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})

    rebuild_calls = []

    def _rebuild(**kwargs):
        rebuild_calls.append(kwargs)
        return disk_catalog

    monkeypatch.setattr(cfg, "get_available_models", _rebuild)

    # Three hits, each 0.1 s apart, simulating back-to-back session visits.
    for i in range(3):
        time.sleep(0.1)
        result = cfg.get_available_models_for_session_visit()
        assert result == disk_catalog

    assert rebuild_calls == [], (
        f"no live rebuild should have run across {3} back-to-back hits, "
        f"but got {len(rebuild_calls)} call(s): {rebuild_calls}"
    )
    # Final mtime should be ~now, not the 290-s-old planted value.
    final_age = time.time() - cache_path.stat().st_mtime
    assert final_age < 2.0, (
        f"final mtime should be re-stamped to ~now after 3 hits, got age={final_age:.3f}s"
    )


# ── 5. test_stale_path_still_triggers_rebuild_unaffected_by_fix ───────────


def test_stale_path_still_triggers_rebuild_unaffected_by_fix(tmp_path, monkeypatch):
    """Guard against an over-eager fix: when the disk mtime is already
    *older* than the freshness window, the response must still fall through
    to the live rebuild path. The re-stamp is only on the hit branch.
    """
    import api.config as cfg

    _reset_models_memory_cache(monkeypatch)
    rebuilt_catalog = _catalog("rebuilt-fresh")
    cache_path = tmp_path / "models_cache.profile.json"
    cache_path.write_text("{}", encoding="utf-8")
    very_old = time.time() - 3600.0  # 1 h old, well past the 300 s window
    os.utime(cache_path, (very_old, very_old))

    monkeypatch.setattr(cfg, "_SESSION_VISIT_MODELS_FRESHNESS_SECONDS", 300.0, raising=False)
    monkeypatch.setattr(cfg, "_get_models_cache_path", lambda: cache_path)
    monkeypatch.setattr(cfg, "_load_models_cache_from_disk", lambda: None)  # simulate _is_loadable_disk_cache rejecting
    monkeypatch.setattr(cfg, "_load_stale_models_cache_from_disk", lambda: None)
    monkeypatch.setattr(cfg, "_models_cache_source_fingerprint", lambda: {"profile": "demo"})

    rebuild_calls = []

    def _rebuild(**kwargs):
        rebuild_calls.append(kwargs)
        return rebuilt_catalog

    monkeypatch.setattr(cfg, "get_available_models", _rebuild)

    result = cfg.get_available_models_for_session_visit()
    assert result == rebuilt_catalog
    assert rebuild_calls == [{"force_refresh": True}]
    # mtime must NOT have been re-stamped on the stale path.
    age_after = time.time() - cache_path.stat().st_mtime
    assert age_after > 3000.0, (
        f"stale path must leave the mtime alone (still ≈1 h old); got age={age_after:.3f}s"
    )
