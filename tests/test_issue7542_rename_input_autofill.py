"""Regression tests for #7542: rename-conversation input must not
trigger the browser's password-save dialog.

Chrome and password-manager extensions (1Password, LastPass, Bitwarden,
Dashlane) mis-classify the chat-title editor as a login form because
it accepts arbitrary user input next to the chat UI. The fix tags
the input with the standard ``autocomplete="off"`` plus the
extension-specific ignore attributes every other credential-shaped
field in the WebUI already uses.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PANELS_JS = (REPO / "static" / "panels.js").read_text(encoding="utf-8")


def _extract_rename_input_setup() -> str:
    """Pull the input setup block from the titlebar double-click
    handler so the test exercises the real code path.
    """
    i = PANELS_JS.find("const inp = document.createElement('input')")
    assert i != -1, "rename input creation not found in panels.js"
    end_marker = "// Prevent click/dblclick"
    end = PANELS_JS.find(end_marker, i)
    assert end != -1
    return PANELS_JS[i:end]


# ── Static-source assertions (cheap, no Node boot) ──────────────────


def test_rename_input_disables_autocomplete():
    """The rename-conversation input must set ``autocomplete='off'``
    so Chrome does not surface the password-save dialog after the
    user renames a chat (#7542).
    """
    block = _extract_rename_input_setup()
    assert re.search(r"inp\.autocomplete\s*=\s*['\"]off['\"]", block), (
        "The rename-conversation input must set ``autocomplete='off'`` "
        "so Chrome does not surface the password-save dialog after "
        "the user renames a chat (#7542)."
    )


def test_rename_input_disables_autocorrect_and_capitalize():
    """Mobile keyboards surface autocorrect / autocapitalize suggestions
    for a chat title, which is wrong: titles are user-chosen labels,
    not sentences. The standard guard is the same pair of attrs the
    WebUI's password fields use.
    """
    block = _extract_rename_input_setup()
    assert re.search(
        r"inp\.setAttribute\(['\"]autocorrect['\"]\s*,\s*['\"]off['\"]\)", block
    )
    assert re.search(
        r"inp\.setAttribute\(['\"]autocapitalize['\"]\s*,\s*['\"]off['\"]\)", block
    )


def test_rename_input_disables_spellcheck():
    """Spellcheck underlines break the visual identity of a chat
    title. The WebUI's other credential-shaped fields set
    spellcheck=false; the rename input should match.
    """
    block = _extract_rename_input_setup()
    assert re.search(
        r"inp\.setAttribute\(['\"]spellcheck['\"]\s*,\s*['\"]false['\"]\)", block
    ), (
        "Spellcheck underlines break the visual identity of a chat "
        "title. The WebUI's other credential-shaped fields set "
        "spellcheck=false; the rename input should match."
    )


@pytest.mark.parametrize(
    "attr",
    [
        "data-1p-ignore",  # 1Password
        "data-lpignore",   # LastPass
        "data-bwignore",   # Bitwarden
        "data-form-type",  # Dashlane / generic form-type hint
    ],
)
def test_rename_input_sets_password_manager_ignore_attrs(attr):
    """The rename input must set every password-manager ignore
    attribute the WebUI's other credential-shaped fields use.
    Without this, the user's local 1Password / LastPass / Bitwarden /
    Dashlane install will surface a "save password" dialog after
    every chat rename — exactly the bug the issue reports.
    """
    block = _extract_rename_input_setup()
    pattern = (
        rf"inp\.setAttribute\(['\"]{re.escape(attr)}['\"]\s*,"
        rf"\s*['\"](?:true|other)['\"]\)"
    )
    assert re.search(pattern, block), (
        f"Rename input must set the {attr!r} attribute so password "
        f"managers do not surface the credential-save dialog. This is "
        f"the same set the WebUI's other credential-shaped fields use."
    )


# ── Negative contract: the fix does not regress the save path ───────


def test_rename_save_path_still_calls_session_rename_endpoint():
    """The fix is purely additive (input attributes); it must not
    touch the save-path code that POSTs to ``/api/session/rename``.
    Regressing that would mean the rename silently no-ops.
    """
    # The save path is the body of ``finish(save)`` further down in
    # the same handler. Look for the explicit endpoint call.
    assert "'/api/session/rename'" in PANELS_JS, (
        "The save path must still POST to /api/session/rename. The "
        "autofill fix is input-attribute only."
    )


def test_rename_input_setup_appears_exactly_once():
    """Regressing to a duplicated input setup would create a second
    input node and double the rename effect. Ensure the
    ``inp.autocomplete = 'off'`` line (and its siblings) appear
    exactly once in the panel.js source.
    """
    autocomplete_lines = re.findall(
        r"inp\.autocomplete\s*=\s*['\"]off['\"]", PANELS_JS
    )
    assert len(autocomplete_lines) == 1, (
        f"inp.autocomplete = 'off' should appear exactly once in "
        f"panels.js (got {len(autocomplete_lines)}). A duplicate would "
        f"double-set the attribute and is almost certainly a copy/"
        f"paste regression."
    )
