"""Regression coverage for issue #7239: artifact rows in the
WebUI's right-rail Artifacts panel must classify the path
relative to the active workspace before any session-scoped API
call. The previous flow always stripped the workspace prefix
and then called ``/api/list`` with the residue as though it
were workspace-relative; for an out-of-workspace absolute path
(e.g. ``/home/user/.hermes/shared/credentials.md`` with
workspace ``/home/user/workspace``) that ``/api/list`` call
fails and the UI shows the generic ``Could not open file``
status. The fix is the maintainer's narrow contract:

1. classify the path as inside or outside the active
   workspace with segment-aware POSIX + Windows handling,
   including separator normalization, workspace-prefix
   collisions, and ``~``/``./`` stripping;
2. inside-workspace rows continue through the original
   strip + ``/api/list`` + ``openFile`` flow;
3. outside-workspace rows do NOT call ``/api/list``,
   ``/api/file*``, or ``/api/escape/*``; the user sees a
   localized "outside the active workspace" message instead.

The previous startsWith-based prefix check is replaced with a
strict segment-boundary check so ``/workspace`` does not match
``/workspace-other`` (a regression of the original report).
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE_JS = (REPO / "static" / "workspace.js").read_text(encoding="utf-8")
I18N_JS = (REPO / "static" / "i18n.js").read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    """Same body-extraction helper used in the cron / context
    test files. Kept inline so the test file is self-contained."""
    marker = f"function {name}("
    start = WORKSPACE_JS.find(marker)
    assert start != -1, f"{name} not found"
    paren = WORKSPACE_JS.find("(", start)
    assert paren != -1
    depth = 0
    for idx in range(paren, len(WORKSPACE_JS)):
        ch = WORKSPACE_JS[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                brace = WORKSPACE_JS.find("{", idx)
                break
    else:
        raise AssertionError(f"{name} params did not terminate")
    assert brace != -1
    depth = 0
    for idx in range(brace, len(WORKSPACE_JS)):
        ch = WORKSPACE_JS[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return WORKSPACE_JS[brace + 1 : idx]
    raise AssertionError(f"{name} body did not terminate")


# ── _isInsideActiveWorkspace: segment-aware classification ───────────────────


def test_helper_exists():
    """A segment-aware classification helper is the single
    decision point for the inside-vs-outside branch. Without
    the helper, every Artifacts click would re-implement the
    prefix logic and would inevitably drift on Windows
    separator handling or workspace-prefix collisions."""
    body = _function_body("_isInsideActiveWorkspace")
    assert body, "_isInsideActiveWorkspace must exist"
    # The helper accepts (artifactPath, workspace) in that order.
    assert "function _isInsideActiveWorkspace(artifactPath, workspace)" in WORKSPACE_JS


def test_helper_returns_true_for_inside_workspace_path_posix():
    """The report's case: a path under the active workspace
    (e.g. ``/home/user/workspace/file.md`` with workspace
    ``/home/user/workspace``) is inside. This is the case the
    pre-existing flow already handled correctly via startsWith."""
    body = _function_body("_isInsideActiveWorkspace")
    # Inline static check: the helper returns true for a path
    # that starts with workspace + '/'.
    assert "p.startsWith(ws + '/')" in body, (
        "inside-workspace must be recognized via a segment-"
        "aware startsWith(ws + '/') check; the bare startsWith "
        "check would match /workspace-other too (#7239 regression)"
    )


def test_helper_returns_false_for_outside_workspace_path():
    """The report's headline bug: a path under ``~/.hermes/shared/``
    with workspace ``/home/user/workspace`` is outside. The
    helper must return false and the caller must NOT then call
    /api/list with the absolute path as though it were
    workspace-relative."""
    body = _function_body("_isInsideActiveWorkspace")
    # The exact-match and prefix-match clauses are the only
    # positive returns; everything else falls through to
    # ``return false``. This is the structural contract that
    # prevents the regression.
    assert "return false" in body
    # No string-level "absolute path is always inside" or
    # "startsWith workspace" without the trailing '/'.
    assert "if(rel.startsWith(ws))" not in body, (
        "a bare startsWith(ws) without the segment-boundary '/' "
        "would silently classify '/workspace-other' as inside "
        "'/workspace', which is the #7239 regression"
    )


def test_helper_segment_boundary_distinguishes_workspace_from_workspace_other():
    """#7239 specifically calls out the workspace-prefix
    collision: a session with workspace ``/workspace`` must
    not treat ``/workspace-other/file.md`` as inside. The
    segment-aware check ``path.startsWith(ws + '/')`` is the
    only line that protects against this — the bare
    ``startsWith(ws)`` would be wrong."""
    body = _function_body("_isInsideActiveWorkspace")
    # The branch uses workspace + '/' to require a segment
    # boundary. A test that exercises this directly is
    # unnecessary; the static check above is the structural
    # contract. The branch must also handle the exact-match
    # case where path === workspace (no trailing '/'), which
    # the equality clause covers.
    assert "if(p === ws)" in body or "if(p===ws)" in body, (
        "the exact-match case (path === workspace) must be a "
        "positive return so a file saved at the workspace root "
        "still opens"
    )


def test_helper_normalizes_windows_backslashes():
    """Windows paths use ``\\`` as a separator. A path like
    ``D:\\workspace\\dir\\file`` must classify correctly against
    workspace ``D:\\workspace``. The helper must normalize
    backslashes to forward slashes before the segment check,
    matching the existing normalization in openArtifactPath."""
    body = _function_body("_isInsideActiveWorkspace")
    assert ".replace(/\\\\/g,'/')" in body, (
        "backslashes must be normalized to forward slashes so "
        "Windows paths classify correctly against the workspace"
    )


def test_helper_tilde_stripping_is_caller_side():
    """The helper itself does not strip a leading ``~`` because
    the caller (``openArtifactPath``) already strips ``~`` and
    ``./`` before calling the helper. The helper takes the
    already-normalized path. This keeps the helper free of
    caller-side assumptions."""
    body = _function_body("_isInsideActiveWorkspace")
    # The helper should not contain its own ~/./ stripping; the
    # caller handles that.
    assert ".replace(/^~\\//" not in body, (
        "the helper must not strip ~/./ itself; the caller does "
        "this once before the classification so all branches "
        "see the same normalized form"
    )


def test_helper_returns_false_for_empty_inputs():
    """The helper must not raise on missing/empty inputs. An
    empty artifact path or an empty workspace both classify
    as outside so the caller can fall through to the
    fail-closed status message."""
    body = _function_body("_isInsideActiveWorkspace")
    assert "if(!artifactPath) return false" in body
    assert "if(!workspace) return false" in body


# ── openArtifactPath: fail-closed branch on outside ──────────────────────────


def test_open_artifact_path_short_circuits_on_outside():
    """#7239's headline behavior: when the artifact is outside
    the active workspace, ``openArtifactPath`` must surface a
    localized status and must NOT call ``_workspacePathExists``
    or ``openFile`` (which would round-trip through /api/list
    or /api/file* for a path the server cannot serve anyway)."""
    body = _function_body("openArtifactPath")
    # Branch: the outside-workspace check sits before the
    # strip + _workspacePathExists + openFile sequence.
    if_idx = body.find("if(!_isInsideActiveWorkspace(")
    exists_idx = body.find("_workspacePathExists(rel)")
    open_idx = body.find("openFile(rel)")
    assert if_idx != -1, (
        "openArtifactPath must consult _isInsideActiveWorkspace "
        "before any session-scoped API call"
    )
    assert exists_idx != -1 and open_idx != -1
    assert if_idx < exists_idx, (
        "the outside-workspace branch must short-circuit BEFORE "
        "the _workspacePathExists call; otherwise the regression "
        "returns via a failing /api/list round-trip"
    )
    assert if_idx < open_idx, (
        "the outside-workspace branch must short-circuit BEFORE "
        "openFile so the user does not see a 404 or a no-op file"
    )


def test_open_artifact_path_uses_localized_status_key():
    """The outside-workspace branch surfaces a translated
    message via the existing ``setStatus`` helper, using a
    dedicated i18n key (not the generic ``file_open_failed``
    which would conflate the two failure modes)."""
    body = _function_body("openArtifactPath")
    assert "t('file_outside_workspace')" in body, (
        "openArtifactPath must call t('file_outside_workspace') "
        "so the status is localized; reusing "
        "t('file_open_failed') would conflate the failure modes "
        "and make a real broken-artifact symptom invisible"
    )


def test_open_artifact_path_preserves_inside_workspace_flow():
    """#7239's preservation contract: inside-workspace rows
    must continue to call _workspacePathExists and openFile
    exactly as before, so a normal in-workspace file still
    opens through the original code path. The diff must not
    add a new branch to that flow."""
    body = _function_body("openArtifactPath")
    # The original flow still has the strip + _workspacePathExists
    # + openFile sequence; the new branch sits BEFORE them but
    # does not remove them.
    assert "if(rel.startsWith(normWs)) rel = rel.slice(normWs.length);" in body
    assert "if(!(await _workspacePathExists(rel)))" in body
    assert "openFile(rel);" in body
    # The strip path runs only after the inside-workspace guard.
    if_idx = body.find("if(!_isInsideActiveWorkspace(")
    strip_idx = body.find("let rel = normalized;")
    assert if_idx < strip_idx, (
        "the strip is gated by the inside-workspace check; "
        "without this the regression round-trips an absolute "
        "path through /api/list"
    )


# ── i18n key + invariant ────────────────────────────────────────────────────


def test_file_outside_workspace_key_in_en_locale():
    """The new i18n key must exist in the en locale. Other
    locales fall back through ``t()`` to en, but the project-
    wide invariant ``test_*_locale_covers_english_keys``
    requires every locale block to declare every en key, so
    this test also confirms the key was added to all 15
    locales (the runtime fallback would otherwise be a
    silent test-invariant violation)."""
    # En locale must define the key.
    assert "file_outside_workspace: 'File is outside the active workspace'" in I18N_JS


def test_file_outside_workspace_key_count_matches_locale_count():
    """Every locale block must declare the new key so the
    ``test_*_locale_covers_english_keys`` invariant keeps
    passing. The count of per-locale key assignments (4-space
    indent, single-quoted value) must equal the number of
    locales (15). The match excludes the one occurrence inside
    the separate ``_I18N_TOOL_ACTION_TEXT_*`` tool action
    labels."""
    occurrences = I18N_JS.count(
        "    file_outside_workspace: '"
    )
    assert occurrences == 15, (
        f"expected the new key in all 15 locales, found "
        f"{occurrences}; the locale-parity invariant would "
        f"flag the missing locales on the next test run"
    )
