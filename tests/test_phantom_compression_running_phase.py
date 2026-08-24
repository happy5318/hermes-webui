"""A bare `running` lifecycle row must not rebuild a phantom "Compressing context" divider.

Reopening a session rebuilds the worklog from persisted activity rows via
``_sourceEventTypeForSnapshotAnchorRow()``. That helper used to classify ANY
lifecycle row whose phase/status was ``running`` as ``'compressing'``, so every
session that left a mid-flight lifecycle row behind (interrupted turn, dropped
terminal event, long provider stall) painted a "Compressing context" barrier on
reopen — even though the backend never compressed anything.

The static completion guard cannot rescue it: the synthetic ``compressed``
backfill (``_ensureAnchorCompressionCompletedOnLiveProgress``) only runs on live
progress, so a session the user merely opens keeps the phantom divider forever.

These tests execute the real helper in node (not a substring assertion) so the
classification contract is verified on the production code path.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MESSAGES_JS_PATH = ROOT / "static" / "messages.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(not NODE, reason="node is required to execute the helper")


_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(MESSAGES_JS, 'utf8');

function extractFunc(name){
  const start = src.indexOf('function ' + name);
  if(start === -1) throw new Error(name + ' not found');
  const params = src.indexOf('(', start);
  let depth = 0, close = -1;
  for(let i = params; i < src.length; i++){
    if(src[i] === '(') depth++;
    else if(src[i] === ')'){ depth--; if(depth === 0){ close = i; break; } }
  }
  if(close === -1) throw new Error(name + ' params did not close');
  const brace = src.indexOf('{', close);
  depth = 0;
  for(let i = brace; i < src.length; i++){
    if(src[i] === '{') depth++;
    else if(src[i] === '}'){
      depth--;
      if(depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error(name + ' body did not close');
}

eval(extractFunc('_sourceEventTypeForSnapshotAnchorRow'));
const classify = _sourceEventTypeForSnapshotAnchorRow;
"""


def _run(body):
    script = (
        "const MESSAGES_JS = " + json.dumps(str(MESSAGES_JS_PATH)) + ";\n"
        + _HARNESS
        + body
    )
    result = subprocess.run([NODE, "-e", script], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _classify(rows):
    """Classify a list of activity rows through the real production helper."""
    body = (
        "const rows = " + json.dumps(rows) + ";\n"
        "process.stdout.write(JSON.stringify(rows.map(r => classify(r))));\n"
    )
    return _run(body)


def test_bare_running_lifecycle_row_is_not_compressing():
    """The regression: a mid-flight lifecycle row must classify as neutral ('')."""
    rows = [
        # phase='running' with no compression text at all
        {"role": "lifecycle", "kind": "lifecycle_status", "phase": "running", "text": ""},
        # status carries 'running' (helper reads phase||status)
        {"role": "lifecycle", "kind": "lifecycle_status", "status": "running", "text": ""},
        # running plus unrelated progress prose
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "phase": "running",
            "text": "Working on request",
        },
        # running plus a fallback/rate-limit notice
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "phase": "running",
            "text": "Rate limited — switching to fallback provider...",
        },
    ]
    assert _classify(rows) == ["", "", "", ""]


def test_genuine_compression_cues_still_classify_as_compressing():
    """The guard must not over-correct: real compression rows still paint."""
    rows = [
        {"role": "lifecycle", "kind": "lifecycle_status", "phase": "compressing", "text": ""},
        {"role": "lifecycle", "kind": "lifecycle_status", "text": "Compressing context"},
        {"role": "lifecycle", "kind": "lifecycle_status", "text": "⟳ Compacting context…"},
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "text": "📦 Pre-API compression: ~217,560 tokens >= 204,800 threshold",
        },
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "text": "🗜️ Context too large (~217,560 tokens) — compressing (1/3)...",
        },
        {"role": "lifecycle", "kind": "lifecycle_status", "text": "compression attempt 2/3"},
    ]
    assert _classify(rows) == ["compressing"] * 6


def test_running_phase_with_real_compression_text_still_compresses():
    """A row that is BOTH running and genuinely compressing must still paint."""
    rows = [
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "phase": "running",
            "text": "Compressing context",
        },
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "status": "running",
            "text": "⟳ Compacting context…",
        },
    ]
    assert _classify(rows) == ["compressing", "compressing"]


def test_completed_compression_rows_still_classify_as_compressed():
    """The 'compressed' branch is untouched by the running-phase narrowing."""
    rows = [
        {"role": "lifecycle", "kind": "lifecycle_status", "phase": "compressed", "text": ""},
        {"role": "lifecycle", "kind": "lifecycle_status", "phase": "done", "text": ""},
        {"role": "lifecycle", "kind": "lifecycle_status", "text": "Context auto-compressed"},
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "text": "🗜️ Compressed 924 → 143 messages, retrying...",
        },
    ]
    assert _classify(rows) == ["compressed"] * 4


def test_bare_preflight_row_is_neutral_but_authoritative_compaction_is_not():
    """Preflight is only a decision log; compaction is the active-start cue."""
    rows = [
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "text": "📦 Preflight compression: ~101,000 tokens >= 96,000 threshold",
        },
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "text": "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        },
    ]
    assert _classify(rows) == ["", "compressing"]


def test_skip_and_defer_notices_never_compress():
    """Skip/cooldown notices must stay neutral even on a running row."""
    rows = [
        {
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "phase": "running",
            "text": "Skipping preflight compression: same-session cooldown active",
        },
    ]
    assert _classify(rows) == [""]


def test_non_lifecycle_roles_are_unaffected():
    """Sanity: the narrowing is scoped to lifecycle rows only."""
    rows = [
        {"role": "prose", "kind": "process_prose"},
        {"role": "thinking", "kind": "reasoning"},
        {"role": "tool", "status": "running"},
        {"role": "tool", "status": "completed"},
        # A running terminal row must not invent a compression start either.
        {"role": "terminal", "kind": "terminal_status", "status": "running"},
        {"role": "terminal", "kind": "terminal_status", "status": "completed"},
    ]
    assert _classify(rows) == ["token", "reasoning", "tool", "tool_complete", "", "done"]


def test_explicit_source_event_type_wins_over_inference():
    """A row that already carries a real source_event_type is passed through."""
    rows = [
        {"source_event_type": "compressing", "role": "lifecycle", "phase": "running"},
        {"source_event_type": "compressed", "role": "lifecycle", "phase": "done"},
        # runtime_journal_snapshot is the sentinel meaning "infer from the row"
        {
            "source_event_type": "runtime_journal_snapshot",
            "role": "lifecycle",
            "kind": "lifecycle_status",
            "phase": "running",
            "text": "",
        },
    ]
    assert _classify(rows) == ["compressing", "compressed", ""]
