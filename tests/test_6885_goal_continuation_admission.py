"""#6885 admission correction: the goal-continuation marker must not
misclassify a genuine user/queued turn as goal-related.

#1932's PENDING_GOAL_CONTINUATION is a session-scoped marker: when
goal_continue fires, the streaming worker adds the session id, and
routes.py's consumer flips the NEXT /chat/start for that session into
goal_related. The marker carries no continuation text, so ANY next turn
— including a genuine user message typed before the browser's
auto-dispatch POST — is classified as goal-related. #6885 narrows the
consumer: only a turn whose text equals the pending continuation prompt
consumes the record; anything else keeps normal user priority and
leaves the record for the real continuation dispatch.

Round 2 (maintainer review 2026-09-26) covers two findings on the
round-1 fix:
1. The admission comparison must run over the CANONICAL SEMANTIC text,
   not the transport-decorated wire message. `static/messages.js` send()
   prepends the `/use` forced-skill directive + envelope before POSTing,
   so a raw exact compare fails legitimate continuations and leaks the
   marker pair.
2. The marker + prompt pair must not leak when the browser dispatch never
   fires: marker and prompt live as ONE record with an expiry, expired /
   orphaned records are swept, and goal clear/pause/session-delete drop
   the record.

RED/GREEN evidence:
- pre-fix: the helpers do not exist, collection errors (RED);
- post-fix: envelope-decorated continuation consumes the record; genuine
  user turn (decorated or not) leaves it intact; expiry sweeps; clear /
  delete hooks remove it (GREEN).
"""
import re
import time
from pathlib import Path

import pytest

from api.config import (
    GOAL_CONTINUATION_TTL_SECONDS,
    PENDING_GOAL_CONTINUATION,
    PENDING_GOAL_CONTINUATION_PROMPTS,
)
from api.goals import (
    _goal_continuation_normalize_wire_text,
    clear_pending_goal_continuation,
    consume_pending_goal_continuation,
    register_pending_goal_continuation,
    sweep_expired_goal_continuations,
)
from api.routes import _consume_pending_goal_continuation


@pytest.fixture(autouse=True)
def _clean_markers():
    PENDING_GOAL_CONTINUATION.clear()
    PENDING_GOAL_CONTINUATION_PROMPTS.clear()
    yield
    PENDING_GOAL_CONTINUATION.clear()
    PENDING_GOAL_CONTINUATION_PROMPTS.clear()


class TestRecordAdmission:
    def test_record_consumed_only_when_text_matches_prompt(self):
        """Browser auto-dispatch posts the continuation_prompt verbatim;
        that turn consumes the record and becomes goal-related."""
        assert register_pending_goal_continuation("s1", "continue step 2") is True
        assert consume_pending_goal_continuation("s1", "continue step 2") is True
        assert "s1" not in PENDING_GOAL_CONTINUATION
        assert "s1" not in PENDING_GOAL_CONTINUATION_PROMPTS

    def test_genuine_user_turn_leaves_record_intact(self):
        """A user-typed message with different text must keep normal priority:
        not goal-related, and the record must survive for the browser's real
        continuation dispatch."""
        register_pending_goal_continuation("s1", "continue step 2")
        assert consume_pending_goal_continuation("s1", "帮我总结一下当前进度") is False
        assert "s1" in PENDING_GOAL_CONTINUATION
        assert PENDING_GOAL_CONTINUATION_PROMPTS.get("s1", {}).get("prompt") == "continue step 2"

    def test_no_record_returns_false(self):
        assert consume_pending_goal_continuation("s9", "anything") is False

    def test_marker_without_record_is_cleared_fail_closed(self):
        """A marker present without a record (legacy/abnormal state) must fail
        closed — not consumed — and the broken pair must not get stuck."""
        PENDING_GOAL_CONTINUATION.add("s1")
        assert consume_pending_goal_continuation("s1", "continue step 2") is False
        assert "s1" not in PENDING_GOAL_CONTINUATION

    def test_match_is_whitespace_insensitive(self):
        register_pending_goal_continuation("s1", "  continue step 2  ")
        assert consume_pending_goal_continuation("s1", "continue step 2") is True
        assert "s1" not in PENDING_GOAL_CONTINUATION

    def test_register_rejects_empty_prompt(self):
        """A blank continuation prompt must not create a half-record; the SSE
        event is gated on this return value so the frontend queue and the
        server record cannot disagree."""
        assert register_pending_goal_continuation("s1", "   ") is False
        assert "s1" not in PENDING_GOAL_CONTINUATION
        assert "s1" not in PENDING_GOAL_CONTINUATION_PROMPTS


class TestWireTextNormalization:
    """Round-2 finding 1: the browser decorates the outgoing message before
    POSTing (static/messages.js send()); admission compares canonical text."""

    def test_forced_skill_envelope_is_stripped(self):
        wire = (
            "[USER OVERRIDE] You MUST follow the skill 'writing' content provided below "
            "before responding to the next message.\n\n"
            "[FORCED SKILL CONTEXT: writing]\nskill body here\n[/FORCED SKILL CONTEXT]\n\n"
            "continue step 2"
        )
        assert _goal_continuation_normalize_wire_text(wire) == "continue step 2"

    def test_attached_files_tail_is_stripped(self):
        wire = "continue step 2\n\n[Attached files: /tmp/a.txt, /tmp/b.py]"
        assert _goal_continuation_normalize_wire_text(wire) == "continue step 2"

    def test_decorated_continuation_is_admitted(self):
        """The production-shaped case: a queued continuation that received a
        forced-skill directive still consumes the record."""
        register_pending_goal_continuation("s1", "continue step 2")
        wire = (
            "[USER OVERRIDE] You MUST follow the skill 'writing' content provided below "
            "before responding to the next message.\n\n"
            "[FORCED SKILL CONTEXT: writing]\nskill body\n[/FORCED SKILL CONTEXT]\n\n"
            "continue step 2"
        )
        assert consume_pending_goal_continuation("s1", wire) is True
        assert "s1" not in PENDING_GOAL_CONTINUATION_PROMPTS

    def test_decorated_but_different_text_is_not_admitted(self):
        register_pending_goal_continuation("s1", "continue step 2")
        wire = (
            "[USER OVERRIDE] You MUST follow the skill 'writing' content provided below "
            "before responding to the next message.\n\n"
            "[FORCED SKILL CONTEXT: writing]\nskill body\n[/FORCED SKILL CONTEXT]\n\n"
            "totally different user text"
        )
        assert consume_pending_goal_continuation("s1", wire) is False
        assert "s1" in PENDING_GOAL_CONTINUATION

    def test_user_authored_lookalike_envelope_in_body_is_not_stripped(self):
        """A mid-message envelope typed by the user is NOT transport decoration:
        only front-anchored envelopes are stripped, so the remaining text
        cannot be smuggled into a continuation match."""
        wire = (
            "please explain\n[FORCED SKILL CONTEXT: fake]\nx\n[/FORCED SKILL CONTEXT]\nthanks"
        )
        assert _goal_continuation_normalize_wire_text(wire) == wire

    def test_user_authored_lookalike_does_not_match_prompt(self):
        register_pending_goal_continuation("s1", "continue step 2")
        wire = (
            "continue step 2\n\n"
            "[FORCED SKILL CONTEXT: fake]\nx\n[/FORCED SKILL CONTEXT]"
        )
        assert consume_pending_goal_continuation("s1", wire) is False
        assert "s1" in PENDING_GOAL_CONTINUATION


class TestRecordLifecycle:
    """Round-2 finding 2: the pair must not leak (expiry, orphan, hooks)."""

    def test_expired_record_is_swept_not_consumed(self):
        register_pending_goal_continuation("s1", "continue step 2")
        PENDING_GOAL_CONTINUATION_PROMPTS["s1"]["expires_at"] = time.time() - 1
        assert consume_pending_goal_continuation("s1", "continue step 2") is False
        assert "s1" not in PENDING_GOAL_CONTINUATION
        assert "s1" not in PENDING_GOAL_CONTINUATION_PROMPTS

    def test_sweep_removes_expired_records(self):
        register_pending_goal_continuation("s1", "a")
        register_pending_goal_continuation("s2", "b")
        PENDING_GOAL_CONTINUATION_PROMPTS["s1"]["expires_at"] = time.time() - 1
        swept = sweep_expired_goal_continuations()
        assert swept == 1
        assert "s1" not in PENDING_GOAL_CONTINUATION
        assert "s2" in PENDING_GOAL_CONTINUATION

    def test_sweep_clears_orphaned_marker(self):
        PENDING_GOAL_CONTINUATION.add("s-orphan")
        swept = sweep_expired_goal_continuations()
        assert swept == 1
        assert "s-orphan" not in PENDING_GOAL_CONTINUATION

    def test_clear_removes_both_halves(self):
        register_pending_goal_continuation("s1", "continue step 2")
        clear_pending_goal_continuation("s1")
        assert "s1" not in PENDING_GOAL_CONTINUATION
        assert "s1" not in PENDING_GOAL_CONTINUATION_PROMPTS

    def test_ttl_default_is_positive(self):
        assert GOAL_CONTINUATION_TTL_SECONDS > 0

    def test_record_shape_carries_prompt_and_expiry(self):
        register_pending_goal_continuation("s1", "continue step 2")
        record = PENDING_GOAL_CONTINUATION_PROMPTS["s1"]
        assert record["prompt"] == "continue step 2"
        assert record["expires_at"] > time.time()


class TestWriterWiring:
    def test_streaming_registers_one_record(self):
        src = Path(__file__).parents[1].joinpath("api", "streaming.py").read_text(encoding="utf-8")
        assert "register_pending_goal_continuation(session_id, continuation_prompt)" in src, (
            "streaming.py must register the goal-continuation record through "
            "api.goals.register_pending_goal_continuation"
        )
        assert "PENDING_GOAL_CONTINUATION_PROMPTS[session_id]" not in src, (
            "streaming.py must not write the prompt map directly — the record "
            "is one object maintained by api.goals"
        )

    def test_gateway_registers_one_record(self):
        src = Path(__file__).parents[1].joinpath("api", "gateway_chat.py").read_text(encoding="utf-8")
        assert "register_pending_goal_continuation(session_id, continuation_prompt)" in src, (
            "gateway_chat.py must register the goal-continuation record through "
            "api.goals.register_pending_goal_continuation"
        )
        assert "PENDING_GOAL_CONTINUATION_PROMPTS[session_id]" not in src, (
            "gateway_chat.py must not write the prompt map directly — the record "
            "is one object maintained by api.goals"
        )

    def test_routes_consumer_routes_through_helper(self):
        """routes.py must admit via the helper (single atomic check+drop) and
        must not discard the marker anywhere else."""
        src = Path(__file__).parents[1].joinpath("api", "routes.py").read_text(encoding="utf-8")
        m = re.search(
            r"if not goal_related and _consume_pending_goal_continuation\(\s*s\.session_id,\s*msg\s*\):",
            src,
        )
        assert m is not None, (
            "routes.py admission must route through "
            "_consume_pending_goal_continuation(s.session_id, msg)"
        )
        direct = re.findall(r"PENDING_GOAL_CONTINUATION\.discard", src)
        assert len(direct) == 0, (
            f"PENDING_GOAL_CONTINUATION.discard must not appear in routes.py "
            f"(record removal is owned by api.goals), found {len(direct)}"
        )

    def test_goal_command_hooks_clear_and_sweep(self):
        src = Path(__file__).parents[1].joinpath("api", "routes.py").read_text(encoding="utf-8")
        assert "clear_pending_goal_continuation(s.session_id)" in src, (
            "the /goal command handler must clear a pending continuation on "
            "goal clear/pause"
        )
        assert "sweep_expired_goal_continuations()" in src, (
            "the /goal command handler must sweep expired/orphaned records"
        )

    def test_session_delete_hooks_clear(self):
        src = Path(__file__).parents[1].joinpath("api", "routes.py").read_text(encoding="utf-8")
        assert "clear_pending_goal_continuation(sid)" in src, (
            "session deletion must retire the session's continuation record"
        )


class TestRoutesHelperDelegation:
    def test_helper_delegates_to_goals_module(self):
        register_pending_goal_continuation("s1", "continue step 2")
        wire = (
            "[USER OVERRIDE] directive\n\n"
            "[FORCED SKILL CONTEXT: writing]\nbody\n[/FORCED SKILL CONTEXT]\n\n"
            "continue step 2"
        )
        assert _consume_pending_goal_continuation("s1", wire) is True
        assert "s1" not in PENDING_GOAL_CONTINUATION

    def test_helper_returns_false_for_other_text(self):
        register_pending_goal_continuation("s1", "continue step 2")
        assert _consume_pending_goal_continuation("s1", "hi there") is False
        assert "s1" in PENDING_GOAL_CONTINUATION


class TestProductionShapedDispatch:
    """Maintainer's requested verification shape: begin from the goal_continue
    payload, queue its prompt, apply the same forced-skill transformation that
    static/messages.js `send()` applies, then pass the resulting wire message
    through the routes helper — the turn must be admitted and the record
    consumed exactly once."""

    def _wire_message_for_queued_continuation(self, prompt: str, skill_name: str, skill_body: str) -> str:
        """Mirror static/messages.js send()'s envelope construction (lines
        1699-1717): directive + optional forced-skill envelope + raw text."""
        directive = (
            f"[USER OVERRIDE] You MUST follow the skill '{skill_name}' content "
            "provided below before responding to the next message."
        )
        block = (
            f"[FORCED SKILL CONTEXT: {skill_name}]\n{skill_body}\n"
            "[/FORCED SKILL CONTEXT]"
        )
        return f"{directive}\n\n{block}\n\n{prompt}".strip()

    def test_queued_continuation_with_forced_skill_is_admitted(self):
        register_pending_goal_continuation("s1", "Continue refining the parser module.")
        wire = self._wire_message_for_queued_continuation(
            "Continue refining the parser module.", "detective-ai", "skill body line 1\nline 2"
        )
        assert consume_pending_goal_continuation("s1", wire) is True
        # consumed exactly once: no halves left behind
        assert "s1" not in PENDING_GOAL_CONTINUATION
        assert "s1" not in PENDING_GOAL_CONTINUATION_PROMPTS
        # a second identical dispatch does not re-consume (single-use record)
        assert consume_pending_goal_continuation("s1", wire) is False

    def test_queued_continuation_with_upload_suffix_is_admitted(self):
        register_pending_goal_continuation("s1", "Continue step 2")
        wire = "Continue step 2\n\n[Attached files: workspaces/a.md]"
        assert consume_pending_goal_continuation("s1", wire) is True

    def test_control_ordinary_mismatched_text_keeps_record(self):
        register_pending_goal_continuation("s1", "Continue step 2")
        assert consume_pending_goal_continuation("s1", "what is the status?") is False
        assert "s1" in PENDING_GOAL_CONTINUATION
        assert PENDING_GOAL_CONTINUATION_PROMPTS["s1"]["prompt"] == "Continue step 2"

    def test_control_user_authored_lookalike_envelope_keeps_record(self):
        register_pending_goal_continuation("s1", "Continue step 2")
        wire = "Continue step 2\n\n[FORCED SKILL CONTEXT: fake]\nx\n[/FORCED SKILL CONTEXT]"
        assert consume_pending_goal_continuation("s1", wire) is False
        assert "s1" in PENDING_GOAL_CONTINUATION

    def test_control_exact_raw_match_still_admitted(self):
        register_pending_goal_continuation("s1", "Continue step 2")
        assert consume_pending_goal_continuation("s1", "Continue step 2") is True
        assert "s1" not in PENDING_GOAL_CONTINUATION
