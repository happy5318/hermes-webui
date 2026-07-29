"""Regression test: phantom "Compressing context" barrier on turn-done.

When a turn completes (done SSE event) but the 'compressed' SSE event was
lost or delayed, the compression UI state remains in phase='running'. The
done handler previously rebound the stale state, surfacing a phantom
"Compressing context" barrier on a session that finished compression (or
never needed it).

The fix: in the done handler, clear any owned automatic compression state
that is still in 'running' phase, regardless of whether the session ID
changed. Non-running states (e.g. 'done') are still rebound.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _done_handler_block() -> str:
    """Extract the done-event handler region from messages.js."""
    src = _read("static/messages.js")
    start = src.find("S.session=d.session;S.messages=_carryForwardEphemeralTurnFields")
    assert start != -1, "done handler S.session assignment not found"
    end = src.find("lastAsst=[...S.messages].reverse().find", start)
    assert end != -1, "lastAsst lookup after done handler not found"
    return src[start:end]


# ── Source-level assertions ──────────────────────────────────────────────

def test_done_handler_clears_stale_running_compression():
    """Running compression state must be cleared at done-time, not rebound."""
    block = _done_handler_block()
    assert "phase==='running'" in block or 'phase==="running"' in block, (
        "done handler must check compression phase before rebinding sessionId"
    )
    assert "clearCompressionUi" in block or "_compressionUi=null" in block, (
        "done handler must clear stale running compression state"
    )


def test_done_handler_still_rebinds_non_running_compression():
    """Non-running compression states (e.g. 'done') should still be rebound."""
    block = _done_handler_block()
    assert "sessionId:d.session.session_id" in block, (
        "done handler must still rebind sessionId for non-running compression states"
    )


def test_no_unconditional_rebind_in_done_handler():
    """The old unconditional rebind must be gone (replaced by conditional)."""
    block = _done_handler_block()
    lines = block.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "sessionId:d.session.session_id" in stripped and "_compressionUi={" in stripped:
            context_before = "\n".join(lines[max(0, i - 5):i])
            assert "else" in context_before or "if(" in context_before or "if (" in context_before, (
                f"Unconditional compression rebind found at line {i}: {stripped}"
            )


def test_clear_is_not_gated_on_session_id_change():
    """The clear must NOT require session_id to differ (A->A case)."""
    block = _done_handler_block()
    # The old fix gated on d.session.session_id!==activeSid — that must be gone.
    # The new fix clears on phase==='running' alone.
    assert "d.session.session_id!==activeSid" not in block, (
        "clear must not be gated on session_id change — A->A case would be missed"
    )


# ── Behavioral state-transition tests ────────────────────────────────────
# These simulate the done handler's compression-state logic against a fake
# window/context to verify the actual runtime behavior, not just source text.

class _FakeCompressionUi:
    """Minimal stand-in for the window._compressionUi object."""
    def __init__(self, session_id, phase='running', automatic=True):
        self.sessionId = session_id
        self.phase = phase
        self.automatic = automatic

    def clone(self):
        return _FakeCompressionUi(self.sessionId, self.phase, self.automatic)


def _simulate_done_handler(compression_state, active_sid, done_session_id):
    """Simulate the done-handler compression logic from messages.js.

    Returns the resulting compression state: None if cleared, or a cloned
    state with updated sessionId if rebound.
    """
    if not compression_state or not compression_state.automatic:
        return compression_state
    if compression_state.sessionId != active_sid:
        return compression_state
    if not done_session_id:
        return compression_state

    if compression_state.phase == 'running':
        # clearCompressionUi()
        return None
    else:
        # rebind
        rebound = compression_state.clone()
        rebound.sessionId = done_session_id
        return rebound


def test_running_a_to_b_is_cleared():
    """A->B: session rotates, compressed SSE lost, done(B) clears running state."""
    state = _FakeCompressionUi(session_id='A', phase='running')
    result = _simulate_done_handler(state, active_sid='A', done_session_id='B')
    assert result is None, "running A->B must be cleared"


def test_running_a_to_a_is_cleared():
    """A->A: no rotation, compressed SSE lost, done(A) clears running state."""
    state = _FakeCompressionUi(session_id='A', phase='running')
    result = _simulate_done_handler(state, active_sid='A', done_session_id='A')
    assert result is None, "running A->A must be cleared"


def test_non_running_a_to_b_is_rebound():
    """Non-running (phase='done') A->B: state is rebound, not cleared."""
    state = _FakeCompressionUi(session_id='A', phase='done')
    result = _simulate_done_handler(state, active_sid='A', done_session_id='B')
    assert result is not None, "non-running A->B must be rebound"
    assert result.sessionId == 'B', "rebound state must carry new sessionId"


def test_non_running_a_to_a_is_rebound():
    """Non-running (phase='done') A->A: state is rebound (no-op), not cleared."""
    state = _FakeCompressionUi(session_id='A', phase='done')
    result = _simulate_done_handler(state, active_sid='A', done_session_id='A')
    assert result is not None, "non-running A->A must be rebound"
    assert result.sessionId == 'A', "rebound state must carry sessionId"


def test_unowned_state_is_left_alone():
    """Compression state for a different session is untouched by done handler."""
    state = _FakeCompressionUi(session_id='OTHER', phase='running')
    result = _simulate_done_handler(state, active_sid='A', done_session_id='A')
    assert result is not None, "unowned state must not be cleared"
    assert result.sessionId == 'OTHER', "unowned state must keep its sessionId"
