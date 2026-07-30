"""Regression test: phantom "Compressing context" barrier on lost compressed SSE.

Verifies that the done handler in static/messages.js clears stale running
compression state when the compressed SSE event is lost or delayed.

Uses static source analysis — no browser, no network, no playwright — so the
target is hermetic under no-egress CI gates and cannot pass by accident when
the compressing handler is short-circuited (the original Playwright test's
fatal flaw).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_messages_js() -> str:
    return (REPO_ROOT / "static" / "messages.js").read_text(encoding="utf-8")


def _done_handler() -> str:
    """Return the full source of the 'done' SSE event handler."""
    js = _read_messages_js()
    start = js.index("source.addEventListener('done',e=>{")
    rest = js[start + 1 :]
    m = re.search(r"\n    source\.addEventListener\('", rest)
    end = start + 1 + (m.start() if m else len(rest))
    return js[start:end]


def _compressing_handler() -> str:
    """Return the full source of the 'compressing' SSE event handler."""
    js = _read_messages_js()
    start = js.index("source.addEventListener('compressing',e=>{")
    rest = js[start + 1 :]
    m = re.search(r"\n    source\.addEventListener\('", rest)
    end = start + 1 + (m.start() if m else len(rest))
    return js[start:end]


def _compression_cleanup_block(done_handler: str) -> str:
    """Extract the compression-UI cleanup block from the done handler.

    The block starts at the outer ``window._compressionUi`` guard and ends at
    the matching closing brace of the ``if (...) { ... } else { ... }`` shape.
    """
    block_start = done_handler.index(
        "window._compressionUi&&window._compressionUi.automatic"
    )
    depth = 0
    for i in range(block_start, len(done_handler)):
        c = done_handler[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return done_handler[block_start : i + 1]
    raise AssertionError("Could not find matching '}' for compression cleanup block")


# ---------------------------------------------------------------------------
# Structural assertions — must exist and be in the right place
# ---------------------------------------------------------------------------


def test_done_handler_contains_compression_cleanup():
    """The done handler must contain the compression-UI cleanup block."""
    done = _done_handler()
    assert "window._compressionUi&&window._compressionUi.automatic" in done
    assert "window._compressionUi.sessionId===activeSid" in done.replace(" ", "")


def test_cleanup_block_is_inside_done_handler():
    """The cleanup must live inside the done handler, not elsewhere."""
    done = _done_handler()
    block = _compression_cleanup_block(done)
    # Assert it really came from done, not from some other handler
    assert block in done


# ---------------------------------------------------------------------------
# Behavioural contract — what the cleanup does for each phase
# ---------------------------------------------------------------------------


def test_running_phase_is_cleared():
    """When phase is 'running' at done-time, the stale barrier must be cleared."""
    block = _compression_cleanup_block(_done_handler())
    running_branch = re.split(r"\}\s*else\s*\{", block)[0]
    assert "phase==='running'" in running_branch.replace(" ", "")
    assert "clearCompressionUi()" in running_branch


def test_non_running_phase_is_rebound():
    """When phase is NOT 'running', sessionId is rebound to the new session."""
    block = _compression_cleanup_block(_done_handler())
    parts = re.split(r"\}\s*else\s*\{", block)
    assert len(parts) >= 2, "Expected if-else shape in compression cleanup block"
    else_branch = parts[1]
    assert "sessionId:d.session.session_id" in else_branch.replace(" ", "")


# ---------------------------------------------------------------------------
# Gating conditions — when cleanup must and must NOT run
# ---------------------------------------------------------------------------


def test_cleanup_not_gated_on_session_id_change():
    """The running-phase clear must NOT require a session rotation (A->A case).

    If the clear were gated on ``d.session.session_id !== activeSid``, a
    same-session turn whose compressed SSE was lost would keep the phantom
    barrier alive forever.  The guard must be ``phase === 'running'`` alone.
    """
    block = _compression_cleanup_block(_done_handler())
    running_branch = block.split("}else{")[0]
    assert "d.session.session_id!==activeSid" not in running_branch.replace(" ", "")
    assert "d.session.session_id === activeSid" not in running_branch.replace(" ", "")
    assert "phase==='running'" in running_branch.replace(" ", "")


def test_cleanup_gated_on_automatic_and_owner():
    """Cleanup must only fire for automatic compression owned by activeSid."""
    block = _compression_cleanup_block(_done_handler())
    assert "window._compressionUi.automatic" in block
    assert "window._compressionUi.sessionId===activeSid" in block.replace(" ", "")


def test_cleanup_gated_on_done_session_exists():
    """Cleanup must only fire when the done event carries a valid session."""
    block = _compression_cleanup_block(_done_handler())
    assert "d.session&&d.session.session_id" in block.replace(" ", "")


# ---------------------------------------------------------------------------
# Negative controls — unrelated/stale-owner scenarios must NOT be touched
# ---------------------------------------------------------------------------


def test_non_owner_compression_is_ignored():
    """If _compressionUi belongs to a different session, the done handler must
    not touch it.  The outer guard ``sessionId===activeSid`` provides this."""
    block = _compression_cleanup_block(_done_handler())
    outer_guard = block.split("{", 1)[0]
    assert "sessionId===activeSid" in outer_guard.replace(" ", "")


def test_manual_compression_is_ignored():
    """Manual (non-automatic) compression must never be auto-cleared by done."""
    block = _compression_cleanup_block(_done_handler())
    outer_guard = block.split("{", 1)[0]
    assert "automatic" in outer_guard


# ---------------------------------------------------------------------------
# Source-level regression proof
# ---------------------------------------------------------------------------


def test_reverting_cleanup_hunk_breaks_tests():
    """Sanity: if the production cleanup block were removed, at least one
    assertion above would fail.  This test pins the block's presence so a
    revert cannot slip through silently."""
    done = _done_handler()
    # The specific production hunk (approx lines 5854-5872) must be present
    assert "phantom" in done.lower() or "compressed SSE" in done
    assert "phase==='running'" in done.replace(" ", "")
    assert "clearCompressionUi()" in done


# ---------------------------------------------------------------------------
# Compressing handler contract — must establish the barrier that done clears
# ---------------------------------------------------------------------------


def test_compressing_handler_checks_session_ownership():
    """The compressing handler must verify S.session.session_id === activeSid
    before setting _compressionUi.  Without this gate the barrier could be set
    for the wrong session, making the done-side clear semantically meaningless."""
    handler = _compressing_handler()
    assert "S.session.session_id!==activeSid" in handler.replace(" ", "")


def test_compressing_handler_sets_running_phase():
    """The compressing handler must set phase:'running' so the done-side
    running-phase detection has something to match against."""
    handler = _compressing_handler()
    assert "phase:'running'" in handler.replace(" ", "")


def test_compressing_handler_sets_automatic_flag():
    """The compressing handler must mark the barrier as automatic so the done
    handler's outer ``automatic`` guard does not short-circuit."""
    handler = _compressing_handler()
    assert "automatic:true" in handler.replace(" ", "")
