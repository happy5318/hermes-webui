"""The state.db read path must keep the steer row's type signal.

A mid-turn steer is stored by the Agent as a typed user row:
``role='user'``, ``display_kind='steer'``, ``content`` = the full
``[OUT-OF-BAND USER MESSAGE ...] ... [/OUT-OF-BAND USER MESSAGE]`` wrapper.
The WebUI's settle scrub (``api/streaming.py::_unwrap_steer_row_oob_marker``)
keys off that exact pair to unwrap the wrapper, and the sidecar path was fixed
in #7610.

The state.db read path dropped it: ``_project_state_db_message`` projects only
the columns named in its caller's ``optional`` list, and neither the canonical
reader (``get_state_db_session_messages``) nor the regeneration helper listed
``display_kind``. So a CLI/gateway/Telegram session read back through state.db
(``session.is_cli_session``, or any session whose sidecar is behind state.db)
reached the UI with the raw wrapper intact, rendered as a normal user bubble —
and by then the scrub could not heal it, because a row in a display list has
already lost its type (#7834).

The contract asserted here: the steer type signal survives the projection in
both callers, and its absence in an older schema degrades quietly.
"""
import sqlite3

from api.models import (
    _project_state_db_message,
    get_state_db_session_messages,
)


OOB_FRAME = (
    "[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered "
    "once at this position; not tool output and not a new delivery when "
    "replayed from conversation history]\nfocus on the failing test\n"
    "[/OUT-OF-BAND USER MESSAGE]"
)


def _steer_row(**overrides):
    row = {
        "role": "user",
        "content": OOB_FRAME,
        "timestamp": 1.0,
        "id": 7,
        "display_kind": "steer",
    }
    row.update(overrides)
    return row


# --- the projection itself ------------------------------------------------


def test_projection_keeps_display_kind_when_the_caller_lists_it():
    """The projection is column-driven: listing the column is what carries it.
    Both callers list it now; this pins that the mechanism still works."""
    msg = _project_state_db_message(
        _steer_row(),
        available={"display_kind"},
        id_col=False,
        optional=("display_kind",),
    )
    assert msg["display_kind"] == "steer"
    assert msg["role"] == "user"
    # Content is projected verbatim — unwrapping is the settle scrub's job,
    # which needs the type signal to fire at all.
    assert msg["content"] == OOB_FRAME


def test_projection_omits_display_kind_when_the_caller_does_not_list_it():
    """Pins the column-driven shape: an unlisted optional column is not
    projected, so the callers' optional lists are the real contract."""
    row = _steer_row()
    msg = _project_state_db_message(
        row, available={"display_kind"}, id_col=False, optional=()
    )
    assert "display_kind" not in msg


def test_projection_tolerates_a_schema_without_the_column():
    """An older state.db has no ``display_kind`` column at all: the projection
    must not fail or invent a value for it."""
    msg = _project_state_db_message(
        {"role": "user", "content": "hello", "timestamp": 2.0, "id": 8},
        available=set(),
        id_col=False,
        optional=("display_kind",),
    )
    assert "display_kind" not in msg
    assert msg["content"] == "hello"


def test_projection_ignores_an_empty_display_kind_value():
    """Empty values are dropped by the projection's own rule — a steer row must
    not become a phantom typed row through a blank column."""
    msg = _project_state_db_message(
        _steer_row(display_kind=""),
        available={"display_kind"},
        id_col=False,
        optional=("display_kind",),
    )
    assert "display_kind" not in msg


# --- the canonical reader (the path the issue reproduces) ------------------


def _state_db_with_steer(tmp_path):
    """A minimal state.db carrying the row shape the Agent writes."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            profile TEXT,
            started_at REAL,
            last_activity_at REAL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL,
            display_kind TEXT,
            active INTEGER
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (session_id, profile, started_at, last_activity_at)"
        " VALUES ('s1', 'default', 1.0, 5.0)"
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, display_kind, active)"
        " VALUES ('s1', 'user', ?, 3.0, 'steer', 1)",
        (OOB_FRAME,),
    )
    conn.commit()
    conn.close()
    return db_path


def _read_messages(monkeypatch, tmp_path, sid="s1", *, steer=True):
    """Read a session's messages through the canonical state.db reader, with the
    store resolved to an isolated temp db."""
    from api import models as models_module

    db_path = tmp_path / ("state.db" if steer else "old-state.db")
    monkeypatch.setattr(models_module, "_active_state_db_path", lambda: db_path)
    result = get_state_db_session_messages(sid)
    # The plain call returns the message list directly; the revision snapshot
    # wraps it.
    if isinstance(result, dict):
        return result["messages"]
    return list(result)


def test_canonical_reader_carries_the_steer_type_signal(tmp_path, monkeypatch):
    """``GET /api/session?...&messages=1`` must return the steer row typed, or
    the UI renders the raw OOB wrapper and the scrub cannot heal it."""
    _state_db_with_steer(tmp_path)

    msgs = _read_messages(monkeypatch, tmp_path)

    assert len(msgs) == 1, msgs
    row = msgs[0]
    assert row["role"] == "user"
    assert row.get("display_kind") == "steer"
    # The wrapper is still present at this layer — the point is that the type
    # signal arrives with it so the scrub can unwrap it downstream.
    assert "[OUT-OF-BAND USER MESSAGE" in row["content"]


def test_canonical_reader_omits_the_type_signal_on_older_schemas(tmp_path, monkeypatch):
    """A state.db without the column must read back cleanly, not error."""
    db_path = tmp_path / "old-state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, started_at REAL);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, content TEXT, timestamp REAL
        );
        """
    )
    conn.execute("INSERT INTO sessions VALUES ('s1', 1.0)")
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp)"
        " VALUES ('s1', 'user', 'plain row', 2.0)"
    )
    conn.commit()
    conn.close()

    msgs = _read_messages(monkeypatch, tmp_path, steer=False)

    assert msgs and "display_kind" not in msgs[0]
    assert msgs[0]["content"] == "plain row"
