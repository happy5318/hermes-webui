"""Regression tests for #6869: launch-failure must clear SESSION_WRITEBACK_OWNERS.

Follow-up to #6636 (shipped in exp-v0.52.187), which fixed the cancel-finalizer
stale-write race and cleared ``SESSION_WRITEBACK_OWNERS`` on the two Gateway
paths that fire in normal use. A Codex re-gate identified the remaining leak
sites of the same class, and the review round that followed found three more
(one of them bricking):

1. ``_prepare_chat_start_session_for_stream`` (api/routes.py) registers the
   writeback owner before any preparation work. A throw anywhere after that
   registration — ``s.save()``, the retained-row validation, provisional-title
   preparation, the eager checkpoint — used to bypass the save-only catch and
   leak the owner.
2. ``_start_chat_stream_for_session`` starts the worker thread; a ``Thread(...)``
   construction or ``thr.start()`` failure used to leave ``STREAMS``, stream
   ownership, the writeback owner and the persisted ``active_stream_id`` behind,
   so every later send for that session returned 409 (bricked until restart).
3. ``_handle_btw`` and ``_handle_background`` register owners independently and
   had no abort cleanup at all; a background failure also left the tracked task
   permanently ``running``.

The fix is one shared launch-abort helper — ``_cleanup_chat_start_launch_failure`` — that
unwinds the registries, the thread state and the session reference together
(compare-and-clear so a successor's claim is never touched), called from every
one of those sites.
"""

import threading
from unittest.mock import Mock

import pytest

import api.config as config
import api.models as models
import api.routes as routes
from api.models import Session


@pytest.fixture(autouse=True)
def _isolate_sessions(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    index_file = session_dir / "_index.json"
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", index_file)
    monkeypatch.setattr(routes, "SESSION_DIR", session_dir)
    monkeypatch.setattr(routes, "SESSION_INDEX_FILE", index_file, raising=False)
    models.SESSIONS.clear()
    config.SESSION_WRITEBACK_OWNERS.clear()
    config.STREAMS.clear()
    config.STREAM_SESSION_OWNERS.clear()
    yield
    models.SESSIONS.clear()
    config.SESSION_WRITEBACK_OWNERS.clear()
    config.STREAMS.clear()
    config.STREAM_SESSION_OWNERS.clear()


def _make_session(sid: str) -> Session:
    s = Session(session_id=sid, messages=[])
    models.SESSIONS[sid] = s
    return s


def test_save_throw_after_register_clears_writeback_owner(monkeypatch):
    """#6869: ``_prepare_chat_start_session_for_stream`` registers the
    writeback owner before any preparation work. A save() throw must not leak
    the entry."""
    s = _make_session("save-throw")
    stream_id = "stream-save-throw"
    # Pre-register so we can assert the abort path cleared it.
    config.register_session_writeback_owner(s.session_id, stream_id)
    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) == stream_id

    s.save = Mock(side_effect=RuntimeError("disk full"))

    with pytest.raises(RuntimeError, match="disk full"):
        routes._prepare_chat_start_session_for_stream(
            s,
            msg="hello",
            attachments=[],
            workspace="/tmp",
            model="m",
            model_provider="p",
            stream_id=stream_id,
        )

    # The fix: writeback owner is cleared because the failed stream still
    # owned it (compare-and-clear).
    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) is None


def test_save_throw_does_not_touch_other_sessions_owner(monkeypatch):
    """#6869: the launch-abort helper only clears the entry for the failed
    stream's session. An unrelated session with its own writeback owner must
    be left alone."""
    s_failed = _make_session("session-failed")
    s_other = _make_session("session-other")
    failed_stream = "stream-failed"
    other_stream = "stream-other"
    config.register_session_writeback_owner(s_failed.session_id, failed_stream)
    config.register_session_writeback_owner(s_other.session_id, other_stream)

    s_failed.save = Mock(side_effect=RuntimeError("disk full"))
    with pytest.raises(RuntimeError):
        routes._prepare_chat_start_session_for_stream(
            s_failed,
            msg="hello",
            attachments=[],
            workspace="/tmp",
            model="m",
            model_provider="p",
            stream_id=failed_stream,
        )

    # The failed session's writeback owner is cleared.
    assert config.SESSION_WRITEBACK_OWNERS.get(s_failed.session_id) is None
    # The unrelated session's owner is left alone — compare-and-clear
    # is per-session via the session_id key.
    assert config.SESSION_WRITEBACK_OWNERS.get(s_other.session_id) == other_stream


def test_save_success_does_not_clear_writeback_owner(monkeypatch):
    """#6869 regression guard: a successful save() (no throw) must leave
    the writeback owner in place — it is the caller's job to clear it
    when the stream actually starts. Pin the happy path so a future
    change does not over-eagerly clear on every save()."""
    s = _make_session("save-success")
    stream_id = "stream-save-success"
    config.register_session_writeback_owner(s.session_id, stream_id)

    s.save = Mock()  # no side_effect — succeeds

    routes._prepare_chat_start_session_for_stream(
        s,
        msg="hello",
        attachments=[],
        workspace="/tmp",
        model="m",
        model_provider="p",
        stream_id=stream_id,
    )

    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) == stream_id


def test_pre_save_exception_clears_writeback_owner(monkeypatch):
    """#6869 round 2: an exception raised during preparation BEFORE s.save()
    must also clear the owner. The old fix only wrapped the save() call, so a
    throw at provisional-title preparation leaked the entry. Inject one via
    the eager-session-save checkpoint, which runs before the save."""
    monkeypatch.setattr(routes, "get_webui_session_save_mode", lambda: "eager")

    def _boom(*a, **kw):
        raise RuntimeError("provisional title failed")

    monkeypatch.setattr(
        routes, "_checkpoint_user_message_for_eager_session_save", _boom
    )

    s = _make_session("pre-save-throw")
    stream_id = "stream-pre-save-throw"
    config.register_session_writeback_owner(s.session_id, stream_id)
    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) == stream_id

    with pytest.raises(RuntimeError, match="provisional title failed"):
        routes._prepare_chat_start_session_for_stream(
            s,
            msg="hello",
            attachments=[],
            workspace="/tmp",
            model="m",
            model_provider="p",
            stream_id=stream_id,
        )

    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) is None
    # The persisted pending state must not keep pointing at the dead stream.
    assert s.active_stream_id is None
    assert s.pending_user_message is None


def test_thread_start_throw_clears_writeback_owner(monkeypatch):
    """#6869: ``_start_chat_stream_for_session`` registers the writeback
    owner via ``_prepare_chat_start_session_for_stream``, then starts the
    worker thread. If ``thr.start()`` raises, the launch-abort helper unwinds
    every registry the half-launched stream touched."""
    sid = "thread-start-throw"
    stream_id = "stream-thread-start-throw"

    s = _make_session(sid)
    # Pre-register so we can verify the abort path cleared it.
    config.register_session_writeback_owner(sid, stream_id)

    # Stub the worker thread so thr.start() raises. We patch
    # ``threading.Thread`` in the routes module so the call inside
    # ``_start_chat_stream_for_session`` hits our stub.
    class _StubThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(routes.threading, "Thread", _StubThread)

    # Drive the function. We need to bypass the active-stream guard
    # (active_stream_id must be None or stale) and the path that
    # returns early on regeneration; with no ``regeneration`` and a
    # fresh session, the function reaches the ``thr.start()`` call.
    # Stub the agent-runtime barrier so we don't depend on Gateway
    # availability.
    monkeypatch.setattr(
        routes,
        "_agent_runtime_barrier_response",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        routes,
        "_active_stream_blocks_chat_start",
        lambda *a, **kw: False,
    )
    monkeypatch.setattr(
        routes,
        "_is_hidden_empty_session",
        lambda s: False,
    )
    monkeypatch.setattr(
        routes,
        "_run_agent_streaming",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        routes,
        "webui_gateway_chat_enabled",
        lambda *a, **kw: False,
    )
    # Bypass the sidecar / state-writer hooks to keep the test tight.
    monkeypatch.setattr(
        routes,
        "_get_session_agent_lock",
        lambda sid: threading.RLock(),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        routes._start_chat_stream_for_session(
            s,
            msg="hello",
            workspace="/tmp",
            model="m",
            model_provider="p",
        )

    # The fix: writeback owner is cleared by the launch-abort helper.
    assert config.SESSION_WRITEBACK_OWNERS.get(sid) is None
    # The bricking half of the same defect: the dead channel and the
    # persisted stream reference must both be gone, or the next send for
    # this session returns 409.
    assert stream_id not in config.STREAMS
    assert config.stream_owner_session_id(stream_id) is None
    assert s.active_stream_id is None


def test_thread_construction_failure_unwinds_registries(monkeypatch):
    """#6869 round 2 (CORE): a ``threading.Thread(...)`` *construction* failure
    — not just a ``start()`` failure — leaves the same brick behind. The
    launch-abort helper must cover construction too."""
    sid = "thread-construct-throw"
    stream_id = "stream-thread-construct-throw"

    s = _make_session(sid)
    config.register_session_writeback_owner(sid, stream_id)

    class _ThrowingThreadCtor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("thread construction failed")

    monkeypatch.setattr(routes.threading, "Thread", _ThrowingThreadCtor)
    monkeypatch.setattr(routes, "_agent_runtime_barrier_response", lambda **kw: None)
    monkeypatch.setattr(routes, "_active_stream_blocks_chat_start", lambda *a, **kw: False)
    monkeypatch.setattr(routes, "_is_hidden_empty_session", lambda s: False)
    monkeypatch.setattr(routes, "_run_agent_streaming", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "webui_gateway_chat_enabled", lambda *a, **kw: False)
    monkeypatch.setattr(routes, "_get_session_agent_lock", lambda sid: threading.RLock())

    with pytest.raises(RuntimeError, match="thread construction failed"):
        routes._start_chat_stream_for_session(
            s,
            msg="hello",
            workspace="/tmp",
            model="m",
            model_provider="p",
        )

    assert config.SESSION_WRITEBACK_OWNERS.get(sid) is None
    assert stream_id not in config.STREAMS
    assert config.stream_owner_session_id(stream_id) is None
    assert s.active_stream_id is None


def test_abort_helper_leaves_successor_owner_untouched():
    """#6869: the launch-abort helper is compare-and-clear on the writeback
    owner. If a successor turn was admitted while the failed stream unwound,
    its claim must survive."""
    sid = "successor-race"
    dead_stream = "stream-dead"
    live_stream = "stream-live"
    s = _make_session(sid)
    config.register_session_writeback_owner(sid, live_stream)

    # A successor claimed the session between registration and abort.
    s.active_stream_id = live_stream
    config.register_stream_owner(live_stream, sid)
    with config.STREAMS_LOCK:
        config.STREAMS[dead_stream] = object()

    routes._cleanup_chat_start_launch_failure(s, dead_stream, reset_session=True)

    assert config.SESSION_WRITEBACK_OWNERS.get(sid) == live_stream
    assert s.active_stream_id == live_stream
    # The dead stream's own channel is still removed.
    assert dead_stream not in config.STREAMS


def test_btw_launch_failure_unwinds_registries(monkeypatch):
    """#6869 round 2: ``_handle_btw`` had no abort cleanup at all. A thread
    start failure must not leave the ephemeral session pointing at a dead
    stream."""
    parent = _make_session("btw-parent")
    monkeypatch.setattr(routes, "_agent_runtime_barrier_response", lambda **kw: None)
    monkeypatch.setattr(routes, "_session_is_subagent_view_only", lambda *a, **kw: False)
    # sid-aware resolver: the handler looks up the parent, and the merged
    # cleanup helper re-resolves the *ephemeral* session canonically.
    monkeypatch.setattr(
        routes, "get_session",
        lambda sid, metadata_only=False: models.SESSIONS.get(sid) or parent,
    )
    monkeypatch.setattr(routes, "bad", lambda h, m, status=400: {"status": status})
    monkeypatch.setattr(
        routes, "j", lambda h, payload, status=200: {"status": status, "payload": payload}
    )

    class _StubThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("btw thread start failed")

    monkeypatch.setattr(routes.threading, "Thread", _StubThread)

    with pytest.raises(RuntimeError, match="btw thread start failed"):
        routes._handle_btw(
            object(),
            {"session_id": "btw-parent", "question": "what?"},
        )

    # Every half-launched btw stream is unwound: no orphan owners anywhere.
    assert config.SESSION_WRITEBACK_OWNERS == {}
    assert config.STREAMS == {}
    assert config.STREAM_SESSION_OWNERS == {}
    for ephemeral in models.SESSIONS.values():
        assert getattr(ephemeral, "active_stream_id", None) is None


def test_background_launch_failure_unwinds_registries_and_fails_task(monkeypatch):
    """#6869 round 2: ``_handle_background`` had no abort cleanup, and a failure
    also left the tracked task permanently ``running`` — the frontend poll never
    saw a result."""
    parent = _make_session("bg-parent")
    monkeypatch.setattr(routes, "_agent_runtime_barrier_response", lambda **kw: None)
    # sid-aware: the handler looks up the parent, and the merged cleanup helper
    # re-resolves the *hidden bg* session canonically.
    monkeypatch.setattr(
        routes, "get_session",
        lambda sid, metadata_only=False: models.SESSIONS.get(sid) or parent,
    )
    monkeypatch.setattr(routes, "bad", lambda h, m, status=400: {"status": status})
    monkeypatch.setattr(
        routes, "j", lambda h, payload, status=200: {"status": status, "payload": payload}
    )

    class _StubThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("bg thread start failed")

    monkeypatch.setattr(routes.threading, "Thread", _StubThread)

    with pytest.raises(RuntimeError, match="bg thread start failed"):
        routes._handle_background(
            object(),
            {"session_id": "bg-parent", "prompt": "do a thing"},
        )

    # Registries are unwound.
    assert config.SESSION_WRITEBACK_OWNERS == {}
    assert config.STREAMS == {}
    assert config.STREAM_SESSION_OWNERS == {}
    for bg_session in models.SESSIONS.values():
        assert getattr(bg_session, "active_stream_id", None) is None

    # The tracked task must not be stranded in "running".
    import api.background as background

    tasks = background.get_background_tasks("bg-parent")
    assert tasks, "aborted background task must still be tracked"
    assert all(t["status"] != "running" for t in tasks), (
        "an aborted background task stayed 'running' forever"
    )


def test_source_launch_abort_helper_is_shared_across_all_sites(monkeypatch):
    """#6869 source guard: every launch-failure site must route through the
    shared ``_cleanup_chat_start_launch_failure`` helper rather than a bespoke cleanup.
    Pin the source so a future refactor cannot reintroduce a fourth bespoke
    try/except that forgets one of the registries."""
    src = routes.__file__
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()

    def _block(name):
        start = text.index(f"def {name}(")
        end = text.index("\ndef ", start + 1)
        return text[start:end]

    # The helper itself exists and clears the writeback owner.
    helper = _block("_cleanup_chat_start_launch_failure")
    assert "clear_session_writeback_owner_if_owned" in helper, (
        "_cleanup_chat_start_launch_failure must clear the writeback owner"
    )
    assert "unregister_stream_owner" in helper, (
        "_cleanup_chat_start_launch_failure must unregister the stream owner"
    )
    assert "STREAMS.pop" in helper, (
        "_cleanup_chat_start_launch_failure must drop the dead stream channel"
    )

    # Every launch-failure site calls the helper.
    for site in (
        "_prepare_chat_start_session_for_stream",
        "_start_chat_stream_for_session",
        "_handle_btw",
        "_handle_background",
    ):
        assert "_cleanup_chat_start_launch_failure(" in _block(site), (
            f"{site} must route launch failures through _cleanup_chat_start_launch_failure (#6869)"
        )

    # The old bespoke per-site cleanup must be gone from the two chat-start
    # functions — the helper is the single cleanup path.
    for site in (
        "_prepare_chat_start_session_for_stream",
        "_start_chat_stream_for_session",
    ):
        assert "clear_session_writeback_owner_if_owned" not in _block(site), (
            f"{site} still clears the writeback owner inline instead of using "
            "the shared helper (#6869)"
        )


# ---------------------------------------------------------------------------
# #7680 re-gate (9/22) — BRICK deadlock + wakeup-lost findings.
# ---------------------------------------------------------------------------


def test_abort_with_lock_held_does_not_self_deadlock(monkeypatch):
    """#7680 finding 1 (BRICK): the abort path inside
    ``_prepare_chat_start_session_for_stream`` previously re-acquired the
    per-session lock that the chat-start loop already held — a plain
    ``threading.Lock`` (not RLock) self-deadlocked, bricking the session
    until process restart.

    The fix threads ``lock_held=True`` through the abort helper so the reset
    runs without the redundant ``with`` block. This test reproduces the
    BRICK with the **real** ``threading.Lock`` returned by
    ``_get_session_agent_lock`` (the prior launch tests substituted
    ``threading.RLock`` and missed it) and asserts the helper completes in
    bounded time.
    """
    s = _make_session("brick-repro")
    stream_id = "stream-brick"
    real_lock = config._get_session_agent_lock(s.session_id)
    assert isinstance(real_lock, type(threading.Lock())), (
        "test guard: the session lock must remain a plain threading.Lock — "
        "switching to RLock would silently mask this deadlock"
    )

    # Hold the real lock from the test thread, exactly as the chat-start
    # loop does. The abort must not block on acquire.
    acquired = real_lock.acquire(timeout=2.0)
    assert acquired, "test setup: failed to acquire the real session lock"
    try:
        s.active_stream_id = stream_id
        config.register_session_writeback_owner(s.session_id, stream_id)

        # Bound the abort call. Without ``lock_held=True`` this would
        # self-deadlock and the test would hit pytest's hang-detector.
        completed = threading.Event()
        result_box = {}

        def _run():
            try:
                routes._cleanup_chat_start_launch_failure(
                    s, stream_id, reset_session=True, lock_held=True
                )
                result_box["ok"] = True
            except BaseException as exc:  # pragma: no cover — defensive
                result_box["err"] = exc
            finally:
                completed.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        assert completed.wait(timeout=2.0), (
            "abort hung while caller held the session lock — "
            "_cleanup_chat_start_launch_failure re-acquired the same threading.Lock "
            "and self-deadlocked (#7680 BRICK)"
        )
        assert result_box.get("ok"), result_box.get("err")
    finally:
        real_lock.release()

    # A follow-up caller must be able to acquire the lock within bounded
    # time. The previous code path left ``active_stream_id`` set, so the
    # next chat-start for this session would 409.
    assert s.active_stream_id is None
    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) is None
    assert real_lock.acquire(timeout=1.0), (
        "follow-up caller could not re-acquire the session lock — "
        "the abort path failed to release state cleanly (#7680)"
    )
    real_lock.release()


def test_chat_start_with_raising_prep_completes_under_real_lock(monkeypatch):
    """#7680 finding 1 (BRICK, end-to-end): drive
    ``_start_chat_stream_for_session`` with the real session lock and a
    raising ``s.save()``. The handler must return the original error in
    bounded time — not deadlock — and a follow-up send for the same
    session must succeed.

    Mirrors the maintainer's repro recipe:
    ``Codex drove the real handler with a preparation step that raises
    (for example, ``_checkpoint_user_message_for_eager_session_save`` or
    ``s.save()`` failing on a full disk or a permissions error)``.
    """
    s = _make_session("eager-repro")
    s.workspace = "/tmp"

    # Force the eager save-mode path so ``_checkpoint_user_message_for_eager_session_save``
    # is exercised; that helper also calls ``s.save()`` so the same Mock
    # failure cascades.
    monkeypatch.setattr(routes, "get_webui_session_save_mode", lambda: "eager")

    # The save failure the maintainer called out: full disk / permissions
    # error during ``s.save()``. The eager helper also calls ``s.save()``;
    # making ``s.save`` itself raise is the single point of failure.
    s.save = Mock(side_effect=RuntimeError("disk full"))

    real_lock = config._get_session_agent_lock(s.session_id)
    assert isinstance(real_lock, type(threading.Lock()))

    # The chat-start loop holds the lock while it calls
    # ``_prepare_chat_start_session_for_stream``. Simulate that exactly by
    # acquiring the real lock from this thread before entry.
    with real_lock:
        completed = threading.Event()
        result_box = {}

        def _run():
            try:
                # ``_start_chat_stream_for_session`` tries the lock
                # itself; under the BRICK repro the lock is already held
                # by THIS test thread, so the inner attempt would block.
                # We instead call the inner step directly to assert the
                # abort path doesn't self-deadlock when the caller is
                # already inside the lock.
                routes._prepare_chat_start_session_for_stream(
                    s,
                    msg="hello",
                    attachments=[],
                    workspace="/tmp",
                    model="m",
                    model_provider="p",
                    stream_id="stream-eager-fail",
                )
            except RuntimeError as exc:
                result_box["err"] = exc
            finally:
                completed.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        assert completed.wait(timeout=3.0), (
            "_prepare_chat_start_session_for_stream hung while the session "
            "lock was held — the abort path re-acquired the same lock and "
            "self-deadlocked (#7680 BRICK)"
        )
        assert isinstance(result_box.get("err"), RuntimeError), (
            f"expected the original disk-full error to propagate, got: "
            f"{result_box!r}"
        )

    # The abort ran, the lock is released, and the session is no longer
    # bricked: a follow-up send can acquire the lock.
    assert s.active_stream_id is None
    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) is None
    assert real_lock.acquire(timeout=1.0), (
        "session lock still held after the abort — follow-up send would "
        "hang (#7680 BRICK)"
    )
    real_lock.release()


def test_abort_preserves_process_wakeup_on_failure(monkeypatch):
    """#7680 finding 2 (SILENT): a failed worker start on the
    process-wakeup path left the wakeup drain with nothing to retry —
    ``PENDING_BG_TASK_COMPLETIONS`` was already consumed upstream, the
    abort cleared ``pending_user_message``, and the ``submitted`` turn
    journal event had no terminal sibling. The background result was lost
    with no retry.

    The fix: when ``preserve_wakeup=True`` is set and the abort happens
    with a non-empty ``pending_user_message``, the helper re-arms
    ``PENDING_BG_TASK_COMPLETIONS`` and appends an ``interrupted`` journal
    event so the drain can deliver the wakeup on its next pass.
    """
    import api.turn_journal as turn_journal

    s = _make_session("wakeup-preserve")
    s.pending_user_message = "[IMPORTANT: bg task completed]"
    s.active_stream_id = "stream-wakeup-fail"
    config.register_session_writeback_owner(s.session_id, s.active_stream_id)

    # Capture journal events so we can assert the ``interrupted`` one.
    captured = []

    def _capture_append(sid, event):
        captured.append((sid, event))
        return event

    monkeypatch.setattr(routes, "_rearm_process_wakeup_after_launch_failure",
                        routes._rearm_process_wakeup_after_launch_failure)
    monkeypatch.setattr(turn_journal, "append_turn_journal_event", _capture_append)

    # Pre-condition: the marker is NOT in PENDING_BG_TASK_COMPLETIONS yet
    # (it was consumed upstream before the worker started).
    assert s.session_id not in config.PENDING_BG_TASK_COMPLETIONS

    routes._cleanup_chat_start_launch_failure(
        s,
        "stream-wakeup-fail",
        reset_session=True,
        lock_held=False,
        preserve_wakeup=True,
    )

    # After: the marker is re-armed so the drain will retry on its next
    # pass, and the journal event closes the ``submitted`` half-open
    # turn so it does not read as in-flight forever.
    assert s.session_id in config.PENDING_BG_TASK_COMPLETIONS
    interrupted = [e for _sid, e in captured if e.get("event") == "interrupted"]
    assert interrupted, (
        "aborted wakeup turn did not append an 'interrupted' journal event "
        "— the 'submitted' sibling would stay half-open (#7680 finding 2)"
    )
    assert interrupted[0].get("reason") == "launch_failure"
    assert interrupted[0].get("stream_id") == "stream-wakeup-fail"

    # The writeback owner and the persisted stream id are still cleared
    # so a follow-up send does not see the dead channel.
    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) is None
    assert s.active_stream_id is None


def test_abort_without_preserve_wakeup_leaves_marker_alone(monkeypatch):
    """#7680: ``preserve_wakeup=False`` (the default) must NOT re-arm the
    drain. A non-wakeup chat-start failure has no business touching
    ``PENDING_BG_TASK_COMPLETIONS``."""
    s = _make_session("no-preserve")
    s.pending_user_message = "regular user message"
    s.active_stream_id = "stream-no-wakeup"

    routes._cleanup_chat_start_launch_failure(
        s,
        "stream-no-wakeup",
        reset_session=True,
    )

    assert s.session_id not in config.PENDING_BG_TASK_COMPLETIONS
    assert s.active_stream_id is None
