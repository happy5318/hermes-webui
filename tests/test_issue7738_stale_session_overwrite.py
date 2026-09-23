"""Regression coverage for issue #7738 — stale evicted Session object can
overwrite a newer save in /api/session/rename (and the same-shape handlers
/api/session/move and /api/session/archive).

Root cause: the handlers resolved the canonical Session object OUTSIDE the
per-session lock. Between the resolve and the lock acquire, count-cap eviction
(``_evict_sessions_over_cap``) could drop that clean, persisted object from
``SESSIONS``. A second writer (e.g. ``/api/session/draft``) would then call
``get_session()``, get a *fresh* object from disk, mutate it and save it.
When the first handler finally took the lock, it saved its detached, stale
object, and every field the second writer changed silently reverted on disk.

The fix mirrors the existing pattern in
``_persist_generated_session_title`` (api/routes.py): re-resolve the canonical
session INSIDE the lock (``SESSIONS.get(sid)`` → ``Session.load(sid)``
fallback → ``_ensure_full_session_before_mutation``), then apply the
mutation and save the FRESH object. The 404 / 403 contracts are still
served from the outside-resolve path so the responses are unchanged.

These tests simulate the race deterministically by monkeypatching
``_get_or_materialize_session`` to return a STALE object and, when the lock
acquire fires, replacing the SESSIONS entry with a NEWER object (mirroring
the eviction + reload that the second writer would do on master). The
postcondition the fix guarantees: the FRESH (newer) object's other-field
mutations survive the rename's save().
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import api.models as models
import api.routes as routes
from api.models import SESSIONS, Session


# ---------------------------------------------------------------------------
# Test helpers (mirrors tests/test_issue2057_worktree_lifecycle.py)
# ---------------------------------------------------------------------------


class _FakeHandler:
    """Minimal BaseHTTPRequestHandler stand-in for handle_post() direct calls.

    handle_post() / bad() / j() only need handler.send_response, send_header,
    end_headers. CSRF + read_body + j are monkeypatched in _capture_post.
    """

    def __init__(self, body: bytes = b"{}"):
        self.headers = {
            "Content-Length": str(len(body)),
            "Content-Type": "application/json",
        }
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.client_address = ("127.0.0.1", 12345)

    def send_response(self, status):  # noqa: D401 — protocol stub
        self.status = status

    def send_header(self, key, value):  # noqa: D401 — protocol stub
        pass

    def end_headers(self):  # noqa: D401 — protocol stub
        pass


def _capture_post(monkeypatch, body):
    """Wire up the standard ``handle_post`` bypass shims and return a recorder.

    Mirrors ``tests/test_issue2057_worktree_lifecycle.py::_capture_post``.
    """
    captured: dict = {}
    monkeypatch.setattr(routes, "_check_csrf", lambda handler: True)
    monkeypatch.setattr(routes, "read_body", lambda handler: body)

    def _j(handler, payload, status=200, extra_headers=None, **_kw):
        captured["payload"] = payload
        captured["status"] = status
        return True

    def _bad(handler, msg, status=400, **_kw):
        captured["payload"] = {"error": msg}
        captured["status"] = status
        return True

    monkeypatch.setattr(routes, "j", _j)
    monkeypatch.setattr(routes, "bad", _bad)
    return captured


def _isolate_session_store(tmp_path, monkeypatch):
    """Point api.models and api.routes at an isolated session dir for the test.

    Mirrors ``tests/test_issue2057_worktree_lifecycle.py::_isolate_session_store``.
    """
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    index_file = session_dir / "_index.json"
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", index_file)
    monkeypatch.setattr(routes, "SESSION_DIR", session_dir)
    monkeypatch.setattr(routes, "SESSION_INDEX_FILE", index_file)
    SESSIONS.clear()
    return session_dir


def _load_disk_session(session_dir, sid):
    """Read the on-disk sidecar for ``sid`` and return the parsed JSON dict."""
    path = session_dir / f"{sid}.json"
    return json.loads(path.read_text(encoding="utf-8"))


class _SwapLock:
    """Proxy around a real ``threading.Lock`` whose ``acquire`` swaps the
    SESSIONS entry for ``swap_with`` just before delegating.

    Mirrors the threading.Lock interface used by the handlers
    (``acquire`` / ``release`` / ``__enter__`` / ``__exit__`` /
    ``locked``) so the ``with _swap_lock:`` form continues to work.

    Why not monkeypatch ``lock.acquire``? ``_thread.lock``'s ``acquire`` is
    a C-implemented read-only slot on Python <3.13. Wrapping the lock
    object is the only way to inject the swap deterministically.
    """

    def __init__(self, inner, sid, swap_with):
        self._inner = inner
        self._sid = sid
        self._swap_with = swap_with

    def acquire(self, *args, **kwargs):
        # Replace the SESSIONS entry just before the lock is taken. This
        # stands in for a concurrent writer having evicted+reloaded+mutated
        # the session in the window between the outside resolve and our
        # lock acquire (the exact race #7738 names).
        SESSIONS[self._sid] = self._swap_with
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        self._inner.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    @property
    def locked(self):
        return self._inner.locked


class _DropLock:
    """Proxy whose ``acquire`` evicts the SESSIONS entry entirely before
    delegating, exercising the Session.load(sid) fallback branch in the
    fix. Used for the "fully evicted, no resident copy" regression test.
    """

    def __init__(self, inner, sid):
        self._inner = inner
        self._sid = sid

    def acquire(self, *args, **kwargs):
        SESSIONS.pop(self._sid, None)
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        self._inner.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    @property
    def locked(self):
        return self._inner.locked


def _install_swap_lock(monkeypatch, sid, swap_with):
    """Replace ``routes._get_session_agent_lock`` with a factory that
    returns a _SwapLock (or _DropLock if ``swap_with is None``) proxy.
    """
    real_lock_factory = routes._get_session_agent_lock

    def _proxy_factory(value):
        lock = real_lock_factory(value)
        if swap_with is None:
            return _DropLock(lock, value)
        return _SwapLock(lock, value, swap_with)

    monkeypatch.setattr(routes, "_get_session_agent_lock", _proxy_factory)


def _stub_post_lock_side_effects(monkeypatch):
    """Stub the rename's post-lock side effects so the test stays focused
    on the stale-overwrite race itself.
    """
    monkeypatch.setattr(routes, "_sync_session_title_to_insights", lambda s: None)
    monkeypatch.setattr(
        routes, "publish_session_list_changed", lambda *a, **kw: None,
    )


# ---------------------------------------------------------------------------
# /api/session/rename — the primary #7738 hot path
# ---------------------------------------------------------------------------


def _seed_stale_session(session_dir, sid, *, title, draft):
    """Persist a baseline on disk matching the stale object so
    Session.load(sid) finds the session, and return a STALE Session object
    the buggy code would resolve. ``stale.save()`` is what writes the
    on-disk baseline.
    """
    stale = Session(
        session_id=sid,
        title=title,
        workspace=str(session_dir.parent),
        messages=[{"role": "user", "content": "hi"}],
        composer_draft=draft,
    )
    stale.save()
    SESSIONS[sid] = stale
    return stale


def _make_newer_session(session_dir, sid, *, title, draft):
    """A fresh object that reflects what a concurrent writer would have
    produced after eviction+reload. Only the fields a different writer
    would mutate differ from the stale object — the rename target's own
    fields are the same.
    """
    return Session(
        session_id=sid,
        title=title,
        workspace=str(session_dir.parent),
        messages=[{"role": "user", "content": "hi"}],
        composer_draft=draft,
    )


def test_rename_handler_re_resolves_session_under_lock_to_avoid_stale_overwrite(
    tmp_path, monkeypatch,
):
    """#7738 primary regression: /api/session/rename must save the resident
    (newer) object, not the stale, evicted one.

    Pre-fix: the outside-resolved stale object was saved; the newer
    composer_draft value was silently clobbered back to the stale value.
    Post-fix: the inside-lock re-resolve picks up the newer object, the
    rename is applied to IT, and its (newer) composer_draft survives.
    """
    session_dir = _isolate_session_store(tmp_path, monkeypatch)
    sid = "stale-rename-1"
    body = {"session_id": sid, "title": "new_title"}
    captured = _capture_post(monkeypatch, body)

    stale = _seed_stale_session(
        session_dir, sid,
        title="old_title",
        draft={"text": "old draft", "files": ["old.txt"]},
    )
    newer = _make_newer_session(
        session_dir, sid,
        title="old_title",
        draft={"text": "new draft", "files": ["new.txt"]},
    )
    monkeypatch.setattr(
        routes, "_get_or_materialize_session", lambda value: stale,
    )
    _install_swap_lock(monkeypatch, sid, swap_with=newer)
    _stub_post_lock_side_effects(monkeypatch)

    result = routes.handle_post(
        _FakeHandler(b'{"session_id": "' + sid.encode() + b'", "title": "new_title"}'),
        SimpleNamespace(path="/api/session/rename"),
    )

    assert result is True, "rename handler must claim the request"
    assert captured["status"] == 200, f"expected 200, got {captured}"
    on_disk = _load_disk_session(session_dir, sid)
    assert on_disk["title"] == "new_title", (
        f"rename must persist the new title; on-disk: {on_disk['title']!r}"
    )
    assert on_disk["composer_draft"] == {"text": "new draft", "files": ["new.txt"]}, (
        "rename save() must not clobber a newer field write by a concurrent "
        "writer — the resident (newer) object's composer_draft must survive "
        "on disk (issue #7738). Got stale-clobbered: "
        f"{on_disk['composer_draft']!r}"
    )


def test_rename_handler_falls_back_to_session_load_when_sessions_cache_empty(
    tmp_path, monkeypatch,
):
    """#7738: when the SESSIONS cache no longer holds the session (fully
    evicted, no in-memory copy), the inside-lock re-resolve must fall back
    to Session.load(sid) and still apply the rename to the freshly-loaded
    object.
    """
    session_dir = _isolate_session_store(tmp_path, monkeypatch)
    sid = "stale-rename-2"
    body = {"session_id": sid, "title": "new_title"}
    captured = _capture_post(monkeypatch, body)

    # Persist baseline so Session.load(sid) has something to find.
    baseline = Session(
        session_id=sid,
        title="baseline_title",
        workspace=str(session_dir.parent),
        messages=[{"role": "user", "content": "hi"}],
        composer_draft={"text": "baseline draft"},
    )
    baseline.save()
    SESSIONS.pop(sid, None)

    # Outside-resolve returns a "ghost" object whose state is intentionally
    # wrong (it must NOT be the source of truth for the save). The
    # fix's inside-lock re-resolve must discard it.
    ghost = Session(
        session_id=sid,
        title="ghost_title",
        workspace=str(session_dir.parent),
        messages=[{"role": "user", "content": "hi"}],
        composer_draft={"text": "ghost draft"},
    )
    monkeypatch.setattr(
        routes, "_get_or_materialize_session", lambda value: ghost,
    )
    # Drop SESSIONS[sid] entirely on lock acquire (swap_with=None is the
    # _DropLock sentinel). The fix must fall back to Session.load(sid).
    _install_swap_lock(monkeypatch, sid, swap_with=None)
    _stub_post_lock_side_effects(monkeypatch)

    result = routes.handle_post(
        _FakeHandler(b'{"session_id": "' + sid.encode() + b'", "title": "new_title"}'),
        SimpleNamespace(path="/api/session/rename"),
    )

    assert result is True
    assert captured["status"] == 200, f"expected 200, got {captured}"
    on_disk = _load_disk_session(session_dir, sid)
    assert on_disk["title"] == "new_title"
    # The Session.load(sid) path must have picked up the baseline draft —
    # the ghost outside-resolved object must NOT have leaked onto disk.
    assert on_disk["composer_draft"] == {"text": "baseline draft"}, (
        "Session.load() fallback must read the real on-disk state, not the "
        "ghost outside-resolved object (issue #7738). Got: "
        f"{on_disk['composer_draft']!r}"
    )


def test_rename_handler_preserves_404_for_unknown_session(tmp_path, monkeypatch):
    """#7738: the 404 contract for an unknown sid must be preserved by the
    outside-resolve pre-validation, even though the lock-held re-resolve
    is now the source of truth for the mutation.
    """
    session_dir = _isolate_session_store(tmp_path, monkeypatch)
    sid = "stale-rename-missing"
    captured = _capture_post(
        monkeypatch, {"session_id": sid, "title": "new_title"},
    )
    monkeypatch.setattr(
        routes, "_get_or_materialize_session",
        lambda value: (_ for _ in ()).throw(KeyError(sid)),
    )

    result = routes.handle_post(
        _FakeHandler(b'{"session_id": "' + sid.encode() + b'", "title": "new_title"}'),
        SimpleNamespace(path="/api/session/rename"),
    )

    assert result is True
    assert captured["status"] == 404, f"expected 404, got {captured}"
    # No session file should have been written.
    assert not (session_dir / f"{sid}.json").exists(), (
        "a 404 must not create a session file on disk"
    )


def test_rename_handler_preserves_403_for_read_only_session(tmp_path, monkeypatch):
    """#7738: the 403 contract for a read-only imported session must be
    preserved. The outside-resolve pre-validation raises PermissionError,
    which the rename handler maps to a 403.
    """
    session_dir = _isolate_session_store(tmp_path, monkeypatch)
    sid = "stale-rename-readonly"
    captured = _capture_post(
        monkeypatch, {"session_id": sid, "title": "new_title"},
    )
    monkeypatch.setattr(
        routes, "_get_or_materialize_session",
        lambda value: (_ for _ in ()).throw(PermissionError("read-only imported session")),
    )

    result = routes.handle_post(
        _FakeHandler(b'{"session_id": "' + sid.encode() + b'", "title": "new_title"}'),
        SimpleNamespace(path="/api/session/rename"),
    )

    assert result is True
    assert captured["status"] == 403, f"expected 403, got {captured}"
    # No session file should have been written (the 403 path returns
    # before the lock; verify the isolated session dir is still empty).
    assert session_dir.exists() and not any(session_dir.iterdir()), (
        "a 403 must not create a session file on disk"
    )


# ---------------------------------------------------------------------------
# /api/session/move — same-shape fix
# ---------------------------------------------------------------------------


def test_move_handler_re_resolves_session_under_lock_to_avoid_stale_overwrite(
    tmp_path, monkeypatch,
):
    """#7738 same-shape regression: /api/session/move must also re-resolve
    under the lock so a stale, evicted object does not clobber a newer save.
    """
    session_dir = _isolate_session_store(tmp_path, monkeypatch)
    sid = "stale-move-1"
    captured = _capture_post(
        monkeypatch, {"session_id": sid, "project_id": None},
    )

    # No projects involved: stub load_projects (and the profile match
    # check) so we don't have to seed a project.
    monkeypatch.setattr(routes, "load_projects", lambda: [])

    stale = _seed_stale_session(
        session_dir, sid,
        title="move title",
        draft={"text": "stale draft (clobbered pre-fix)"},
    )
    newer = _make_newer_session(
        session_dir, sid,
        title="move title",
        draft={"text": "newer draft (must survive)"},
    )
    monkeypatch.setattr(
        routes, "_get_or_materialize_session", lambda value: stale,
    )
    _install_swap_lock(monkeypatch, sid, swap_with=newer)
    monkeypatch.setattr(
        routes, "publish_session_list_changed", lambda *a, **kw: None,
    )

    result = routes.handle_post(
        _FakeHandler(b'{"session_id": "' + sid.encode() + b'"}'),
        SimpleNamespace(path="/api/session/move"),
    )

    assert result is True, "move handler must claim the request"
    assert captured["status"] == 200, f"expected 200, got {captured}"
    on_disk = _load_disk_session(session_dir, sid)
    # The newer's composer_draft must have survived the move's save().
    assert on_disk["composer_draft"] == {"text": "newer draft (must survive)"}, (
        "move save() must not clobber a newer field write — the resident "
        "(newer) object's composer_draft must survive on disk (issue #7738). "
        f"Got stale-clobbered: {on_disk['composer_draft']!r}"
    )


# ---------------------------------------------------------------------------
# /api/session/archive — same-shape fix
# ---------------------------------------------------------------------------


def test_archive_handler_re_resolves_session_under_lock_to_avoid_stale_overwrite(
    tmp_path, monkeypatch,
):
    """#7738 same-shape regression: /api/session/archive must also re-resolve
    under the lock so the archived state goes onto the resident (newer)
    object, not the stale one. The materialize-fallback path is not
    exercised here — this is the common in-cache path.
    """
    session_dir = _isolate_session_store(tmp_path, monkeypatch)
    sid = "stale-archive-1"
    captured = _capture_post(
        monkeypatch, {"session_id": sid, "archived": True},
    )
    # Stub the subagent-view-only gate so we don't have to wire state.db.
    monkeypatch.setattr(
        routes, "_session_is_subagent_view_only", lambda value: False,
    )

    stale = _seed_stale_session(
        session_dir, sid,
        title="archive title",
        draft={"text": "stale draft (clobbered pre-fix)"},
    )
    newer = _make_newer_session(
        session_dir, sid,
        title="archive title",
        draft={"text": "newer draft (must survive)"},
    )

    # The archive handler does NOT call _get_or_materialize_session; it
    # uses get_session directly. We hook _get_session_agent_lock instead
    # so the swap fires when the lock is taken.
    def _get_session(sid, **_kwargs):
        return stale

    monkeypatch.setattr(models, "get_session", _get_session)
    monkeypatch.setattr(routes, "get_session", _get_session)
    _install_swap_lock(monkeypatch, sid, swap_with=newer)
    monkeypatch.setattr(
        routes, "publish_session_list_changed", lambda *a, **kw: None,
    )
    # _worktree_retained_payload reads from the session; not relevant for
    # this test. Stub it to {}.
    monkeypatch.setattr(routes, "_worktree_retained_payload", lambda s: {})

    result = routes.handle_post(
        _FakeHandler(b'{"session_id": "' + sid.encode() + b'", "archived": true}'),
        SimpleNamespace(path="/api/session/archive"),
    )

    assert result is True, "archive handler must claim the request"
    assert captured["status"] == 200, f"expected 200, got {captured}"
    on_disk = _load_disk_session(session_dir, sid)
    assert on_disk["archived"] is True, (
        f"archive must persist the new archived=True state; on-disk: {on_disk.get('archived')!r}"
    )
    # The newer's composer_draft must have survived the archive's save().
    assert on_disk["composer_draft"] == {"text": "newer draft (must survive)"}, (
        "archive save() must not clobber a newer field write — the resident "
        "(newer) object's composer_draft must survive on disk (issue #7738). "
        f"Got stale-clobbered: {on_disk['composer_draft']!r}"
    )
