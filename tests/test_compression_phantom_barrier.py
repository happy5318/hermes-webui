"""Regression test: phantom "Compressing context" barrier on session switch.

When a turn completes (done SSE event) but the 'compressed' SSE event was lost
or delayed, the compression UI state remains in phase='running'. The done
handler previously unconditionally rebound the compression state's sessionId to
the continuation session, surfacing a phantom "Compressing context" barrier on
a session that never actually compressed.

The fix: when the done event's session_id differs from the active session AND
the compression UI is still in 'running' phase, clear it instead of rebinding.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def _done_handler_block() -> str:
    """Extract the done-event handler region from messages.js."""
    src = _read("static/messages.js")
    # The compression-rebind logic lives inside the done handler, after
    # S.session=d.session assignment.
    start = src.find("S.session=d.session;S.messages=_carryForwardEphemeralTurnFields")
    assert start != -1, "done handler S.session assignment not found"
    # Grab enough context to cover the compression rebind block.
    end = src.find("lastAsst=[...S.messages].reverse().find", start)
    assert end != -1, "lastAsst lookup after done handler not found"
    return src[start:end]


def test_done_handler_clears_stale_running_compression_on_session_change():
    """When session_id changes and phase is 'running', clear instead of rebind."""
    block = _done_handler_block()
    # The fix must check for phase==='running' before rebinding.
    assert "phase==='running'" in block or 'phase==="running"' in block, (
        "done handler must check compression phase before rebinding sessionId"
    )
    # The fix must call clearCompressionUi or null out _compressionUi.
    assert "clearCompressionUi" in block or "_compressionUi=null" in block, (
        "done handler must clear stale running compression state"
    )


def test_done_handler_still_rebinds_non_running_compression():
    """Non-running compression states (e.g. 'done') should still be rebound."""
    block = _done_handler_block()
    # The else branch must still rebind for non-running states.
    assert "sessionId:d.session.session_id" in block or 'sessionId:d.session.session_id' in block, (
        "done handler must still rebind sessionId for non-running compression states"
    )


def test_no_unconditional_rebind_in_done_handler():
    """The old unconditional rebind must be gone (replaced by conditional)."""
    block = _done_handler_block()
    # The old code was a single unconditional assignment:
    #   window._compressionUi={...window._compressionUi, sessionId:d.session.session_id};
    # After the fix, this line must be inside an else branch, not standalone.
    lines = block.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "sessionId:d.session.session_id" in stripped and "_compressionUi={" in stripped:
            # This line must NOT be the only statement in the if-block.
            # Check that there's an 'if' guard before it or an 'else' on same/prev line.
            context_before = "\n".join(lines[max(0, i - 5):i])
            assert "else" in context_before or "if(" in context_before or "if (" in context_before, (
                f"Unconditional compression rebind found at line {i}: {stripped}"
            )
