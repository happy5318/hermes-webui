"""Regression tests for #6869: launch-failure must clear SESSION_WRITEBACK_OWNERS.

Follow-up to #6636 (shipped in exp-v0.52.187), which fixed the cancel-finalizer
stale-write race and cleared ``SESSION_WRITEBACK_OWNERS`` on the two Gateway
paths that fire in normal use. A Codex re-gate identified one remaining leak
site of the same class on the rare launch-failure path:

- ``_prepare_chat_start_session_for_stream`` (api/routes.py:~22747) registers
  the writeback owner (line 22780) before any thread start. A ``s.save()``
  throw in that function (line 22840) would otherwise leak the entry.
- ``_start_chat_stream_for_session`` (api/routes.py:~23244) starts the
  worker thread; if ``thr.start()`` raises, the existing launch-abort
  cleanup covered the Gateway lifecycle markers but not the per-session
  writeback owner.

The fix is to use ``clear_session_writeback_owner_if_owned()`` (compare-and-clear
so a successor's claim is untouched) on both abort branches. Re-raise so the
caller can complete the rest of the cleanup.
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
    yield
    models.SESSIONS.clear()
    config.SESSION_WRITEBACK_OWNERS.clear()


def _make_session(sid: str) -> Session:
    s = Session(session_id=sid, messages=[])
    models.SESSIONS[sid] = s
    return s


def test_save_throw_after_register_clears_writeback_owner(monkeypatch):
    """#6869: ``_prepare_chat_start_session_for_stream`` registers the
    writeback owner (line 22780) before ``s.save()`` (line 22840). A save()
    throw must not leak the entry."""
    s = _make_session("save-throw")
    stream_id = "stream-save-throw"
    # Pre-register so we can assert the abort path cleared it.
    config.register_session_writeback_owner(s.session_id, stream_id)
    assert config.SESSION_WRITEBACK_OWNERS.get(s.session_id) == stream_id

    s.save = Mock(side_effect=RuntimeError("disk full"))

    # ``defer_save=False`` is the path that calls s.save() unconditionally;
    # ``_resolve_msg_for_request`` etc. are not exercised by this helper.
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
    """#6869: ``clear_session_writeback_owner_if_owned`` only clears
    the entry for the failed stream's session. An unrelated session
    with its own writeback owner must be left alone."""
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


def test_thread_start_throw_clears_writeback_owner(monkeypatch):
    """#6869: ``_start_chat_stream_for_session`` registers the writeback
    owner via ``_prepare_chat_start_session_for_stream``, then starts the
    worker thread. If ``thr.start()`` raises, the existing launch-abort
    cleanup covers the Gateway lifecycle markers but not the per-session
    writeback owner. The fix adds the compare-and-clear in the same
    except branch."""
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

    # The fix: writeback owner is cleared by the thr.start() except
    # branch.
    assert config.SESSION_WRITEBACK_OWNERS.get(sid) is None


def test_source_launch_abort_clears_writeback_owner(monkeypatch):
    """#6869 source guard: both abort branches in api/routes.py
    (save() throw inside _prepare_chat_start_session_for_stream, and
    thr.start() throw inside _start_chat_stream_for_session) must
    call ``clear_session_writeback_owner_if_owned``. Pin the source
    so a future refactor cannot move the cleanup to a different
    branch and silently reintroduce the leak."""
    src = routes.__file__
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()
    # The two anchor functions are unique.
    prep_block_start = text.index("def _prepare_chat_start_session_for_stream(")
    prep_block_end = text.index("\ndef ", prep_block_start + 1)
    prep_block = text[prep_block_start:prep_block_end]
    assert "clear_session_writeback_owner_if_owned" in prep_block, (
        "_prepare_chat_start_session_for_stream must clear the writeback "
        "owner on the s.save() abort path (#6869)"
    )
    start_block_start = text.index("def _start_chat_stream_for_session(")
    start_block_end = text.index("\ndef ", start_block_start + 1)
    start_block = text[start_block_start:start_block_end]
    assert "clear_session_writeback_owner_if_owned" in start_block, (
        "_start_chat_stream_for_session must clear the writeback owner "
        "on the thr.start() abort path (#6869)"
    )
