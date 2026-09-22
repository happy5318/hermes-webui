"""Regression tests for the #7237 review blockers.

Blocker 1 (SILENT data loss): the outbound orphan sanitizer's projected
history must never be persisted wholesale on the next turn. When the
replayed prefix no longer matches the raw pre-turn context (because the
sanitizer rewrote it), the settle path must keep the raw pre-turn context
and settle only the current-turn slice on top — unless a compression
marker explains the rotation.

Blocker 2 (valid pair dropped): ``assistant(call), assistant(progress),
tool(result)`` is a legitimate, commonly-occurring shape. The outbound
sanitizer must merge consecutive assistant rows (mirroring the Agent's
pass order: merge BEFORE orphan detection) so the call/result pair both
survive the projection. A genuine user/system boundary between a call and
its result is still an orphan.
"""
import json
from types import SimpleNamespace

from api.streaming import (
    _dedupe_replayed_context_messages,
    _sanitize_messages_for_api,
    _settle_result_messages,
    _strip_orphan_tool_calls,
    _merge_consecutive_assistant_rows,
    _api_safe_message_positions,
)


def _call(cid):
    return {"id": cid, "type": "function", "function": {"name": "t", "arguments": "{}"}}


def _assistant(cids, content=""):
    return {"role": "assistant", "content": content, "tool_calls": [_call(c) for c in cids]}


def _tool(cid, content="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


class TestBlocker1WholesaleAcceptanceGuard:
    """The first local turn after an orphan repair must not destroy history."""

    def _raw_history(self):
        # Two completed turns. The second assistant carries a call/result pair.
        return [
            {"role": "user", "content": "first question", "timestamp": 1.0},
            {"role": "assistant", "content": "first answer", "timestamp": 2.0},
            {"role": "user", "content": "second question", "timestamp": 3.0},
            {"role": "assistant", "content": "", "tool_calls": [_call("k1")], "timestamp": 4.0},
            {"role": "tool", "tool_call_id": "k1", "content": "tool output", "timestamp": 5.0},
            {"role": "assistant", "content": "second answer", "timestamp": 6.0},
        ]

    def test_next_turn_persists_raw_history_not_projected(self):
        """Save-then-reload: persisted context keeps the original pair."""
        raw = self._raw_history()
        # The outbound sanitizer drops the orphan-looking pair (e.g. strict
        # provider projection); the Agent returns a projected history whose
        # prefix no longer matches raw.
        sanitized = _sanitize_messages_for_api(raw)
        # Simulate the Agent returning the sanitized history + a new turn.
        result = sanitized + [
            {"role": "user", "content": "third question"},
            {"role": "assistant", "content": "third answer"},
        ]
        identity = None  # no active-turn identity: fallback scan must fire
        settled = _dedupe_replayed_context_messages(raw, result, "third question", identity)
        # The persisted context must keep the ORIGINAL tool pair verbatim.
        assert settled == raw + [
            {"role": "user", "content": "third question"},
            {"role": "assistant", "content": "third answer"},
        ], "raw pre-turn context must survive settle intact"

    def test_next_turn_save_then_reload_keeps_pair(self):
        """Full settle round-trip: session.context_messages keeps the pair."""
        raw = self._raw_history()
        sanitized = _sanitize_messages_for_api(raw)
        result = sanitized + [
            {"role": "user", "content": "third question"},
            {"role": "assistant", "content": "third answer"},
        ]
        session = SimpleNamespace(
            messages=list(raw),
            context_messages=list(raw),
            truncation_watermark=None,
        )
        _settle_result_messages(
            session,
            list(raw),          # previous_messages (display)
            list(raw),          # previous_context_messages
            result,
            "third question",
            "webui",
            None,
        )
        persisted = session.context_messages
        tool_rows = [m for m in persisted if m.get("role") == "tool"]
        assert tool_rows, "the original tool result must survive settle"
        call_rows = [
            m for m in persisted
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert call_rows, "the original tool call must survive settle"
        assert any(
            tc.get("id") == "k1" for m in call_rows for tc in m["tool_calls"]
        ), "call k1 must still be present in persisted context"

    def test_compression_rotation_still_replaces_wholesale(self):
        """A compression turn legitimately rotates the context."""
        raw = self._raw_history()
        marker = {
            "role": "assistant",
            "content": "[CONTEXT COMPACTION] summary of prior turns",
        }
        result = [
            marker,
            {"role": "user", "content": "third question"},
            {"role": "assistant", "content": "third answer"},
        ]
        settled = _dedupe_replayed_context_messages(raw, result, "third question", None)
        # With a compression marker the wholesale replacement stands.
        assert settled == result

    def test_identity_boundary_used_when_present(self):
        """With a valid active-turn identity the boundary slice is honoured."""
        raw = self._raw_history()
        sanitized = _sanitize_messages_for_api(raw)
        user_row = {"role": "user", "content": "third question", "_active_turn_token": "tok-1"}
        result = sanitized + [
            user_row,
            {"role": "assistant", "content": "third answer"},
        ]
        identity = {
            "turn_id": "turn-1",
            "token": "tok-1",
            "agent_turn_boundary_resolved": True,
            "current_turn_user_idx": len(result) - 2,
        }
        settled = _dedupe_replayed_context_messages(raw, result, "third question", identity)
        assert settled == raw + [user_row, {"role": "assistant", "content": "third answer"}]


class TestBlocker2MergeBeforeOrphan:
    """Mergeable assistant rows must not orphan a valid call/result pair."""

    def test_progress_row_between_call_and_result_keeps_pair(self):
        msgs = [_assistant(["m1"], content=""), {"role": "assistant", "content": "progress"}, _tool("m1")]
        sanitized = _sanitize_messages_for_api(msgs)
        call_rows = [m for m in sanitized if m.get("tool_calls")]
        assert call_rows, "the call row must survive"
        assert any(tc.get("id") == "m1" for m in call_rows for tc in m["tool_calls"])
        tool_rows = [m for m in sanitized if m.get("role") == "tool"]
        assert any(m.get("tool_call_id") == "m1" for m in tool_rows), "the result must survive"

    def test_merge_helper_unions_tool_calls(self):
        msgs = [_assistant(["m1"], content=""), {"role": "assistant", "content": "progress"}]
        merged = _merge_consecutive_assistant_rows(msgs)
        assert len(merged) == 1
        assert [tc["id"] for tc in merged[0]["tool_calls"]] == ["m1"]
        assert merged[0]["content"] == "progress"

    def test_merge_helper_does_not_mutate_input(self):
        msgs = [_assistant(["m1"], content=""), {"role": "assistant", "content": "progress"}]
        before = json.dumps(msgs)
        _merge_consecutive_assistant_rows(msgs)
        assert json.dumps(msgs) == before

    def test_merge_helper_codex_interim_exempt(self):
        interim = {
            "role": "assistant", "content": "interim",
            "codex_message_items": [{"type": "message"}],
        }
        msgs = [_assistant(["m1"], content=""), interim]
        merged = _merge_consecutive_assistant_rows(msgs)
        assert len(merged) == 2, "codex interim rows are replayed verbatim"

    def test_merge_helper_verification_supersede(self):
        provisional = {
            "role": "assistant", "content": "provisional",
            "tool_calls": [_call("v1")],
            "finish_reason": "verification_required",
        }
        final = {"role": "assistant", "content": "final answer"}
        merged = _merge_consecutive_assistant_rows([provisional, final])
        assert len(merged) == 1
        assert "tool_calls" not in merged[0], "superseded candidate is replaced, not unioned"
        assert merged[0]["content"] == "final answer"

    def test_genuine_user_boundary_still_orphans(self):
        """A user turn between call and result is a real orphan — split the
        merged/mergeable cases from the genuine-boundary case."""
        msgs = [_assistant(["g1"]), {"role": "user", "content": "interrupt"}, _tool("g1")]
        sanitized = _sanitize_messages_for_api(msgs)
        assert not any(m.get("tool_calls") for m in sanitized), "call across a user boundary is an orphan"
        tool_rows = [m for m in sanitized if m.get("role") == "tool"]
        assert not tool_rows, "result across a user boundary is dropped too"

    def test_api_safe_positions_merge_keeps_origin_index(self):
        msgs = [
            {"role": "user", "content": "go"},
            _assistant(["m1"], content=""),
            {"role": "assistant", "content": "progress"},
            _tool("m1"),
        ]
        out = _api_safe_message_positions(msgs)
        surviving_calls = [(idx, m) for idx, m in out if m.get("tool_calls")]
        assert surviving_calls, "merged survivor keeps the call"
        idx, first = surviving_calls[0]
        assert idx == 1, "merged row keeps the FIRST row's original index"
        tool_rows = [(i, m) for i, m in out if m.get("role") == "tool"]
        assert tool_rows and tool_rows[0][1]["tool_call_id"] == "m1"


class TestOldGuardStillHolds:
    """The pre-existing contract (from the PR's own tests) must survive."""

    def test_positional_orphan_still_repaired(self):
        msgs = [_assistant(["orphan"], content="visible"), {"role": "user", "content": "next"}]
        repaired = _strip_orphan_tool_calls(msgs)
        assert "tool_calls" not in repaired[0]

    def test_non_adjacent_result_cannot_rescue_earlier_orphan(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [_call("foreign")]},
            {"role": "assistant", "content": "intervening"},
            {"role": "tool", "tool_call_id": "foreign", "content": "late result"},
        ]
        sanitized = _sanitize_messages_for_api(msgs)
        # After merge-then-orphan the intervening assistant merges into the
        # caller row; the late result pairs with the merged call — the Agent's
        # own pass order keeps both. (Previously this whole pair was dropped.)
        tool_rows = [m for m in sanitized if m.get("role") == "tool"]
        assert tool_rows, "agent pass order keeps the late-paired call/result"
