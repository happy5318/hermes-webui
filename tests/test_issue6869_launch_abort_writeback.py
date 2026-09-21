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

The fix is one shared launch-abort helper — ``_abort_launched_stream`` — that
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

    routes._abort_launched_stream(s, dead_stream, reset_session=True)

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
    monkeypatch.setattr(routes, "get_session", lambda *a, **kw: parent)
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
    monkeypatch.setattr(routes, "get_session", lambda *a, **kw: parent)
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
    shared ``_abort_launched_stream`` helper rather than a bespoke cleanup.
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
    helper = _block("_abort_launched_stream")
    assert "clear_session_writeback_owner_if_owned" in helper, (
        "_abort_launched_stream must clear the writeback owner"
    )
    assert "unregister_stream_owner" in helper, (
        "_abort_launched_stream must unregister the stream owner"
    )
    assert "STREAMS.pop" in helper, (
        "_abort_launched_stream must drop the dead stream channel"
    )

    # Every launch-failure site calls the helper.
    for site in (
        "_prepare_chat_start_session_for_stream",
        "_start_chat_stream_for_session",
        "_handle_btw",
        "_handle_background",
    ):
        assert "_abort_launched_stream(" in _block(site), (
            f"{site} must route launch failures through _abort_launched_stream (#6869)"
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
