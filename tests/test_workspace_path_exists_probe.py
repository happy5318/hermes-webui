"""Regression coverage for the workspace file-existence probe.

The browser answered "does this file exist?" by scanning the parent
directory's listing. That listing is bounded — the server caps a single
``/api/list`` response — so in a workspace with more entries than the cap the
tail of the sorted directory is dropped. A file past the cap is then absent
from the listing even though it is present on disk, and the click handler
concluded "file open failed" for a file that was sitting right there.

This module covers the frontend half of the fix: when the parent listing does
not contain the name, the check asks the server for the file itself instead of
treating the listing's silence as proof of absence.

The server-side pagination for #6645 (#6685, #6687) is a separate, larger
change; this probe deliberately does not depend on any pagination metadata, so
it stays correct on its own and remains a valid fallback once pagination lands.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_JS = (ROOT / "static" / "workspace.js").read_text(encoding="utf-8")


def _node_env() -> dict:
    # node aborts with SIGABRT when the inherited environment carries
    # NODE_CHANNEL_FD, so the checks below run with that variable stripped.
    return {k: v for k, v in os.environ.items() if k != "NODE_CHANNEL_FD"}


def test_probe_is_used_when_the_listing_misses():
    """A name missing from the parent listing must trigger the direct probe.

    The listing may be capped, so its silence is not proof of absence.
    """
    source = WORKSPACE_JS
    exists_fn = source[ source.index("async function _workspacePathExists(") : ]
    exists_fn = exists_fn[: exists_fn.index("\n}\n")]

    assert "_workspaceProbeFile" in exists_fn
    # The listing hit short-circuits; the miss falls through to the probe.
    assert re.search(r"if\s*\(\s*hit\s*\)\s*return true", exists_fn)
    assert "_workspaceProbeFile" in exists_fn.split("if(hit)")[1]
    # A listing error must also fall back rather than report "missing".
    assert "catch" in exists_fn
    assert "_workspaceProbeFile" in exists_fn.split("catch", 1)[1]


def test_probe_uses_the_raw_route_not_the_size_capped_read_route():
    """`read` answers 404 for an over-sized file (MAX_FILE_BYTES), which would
    make a perfectly real (large) file look absent. The probe must use `raw`."""
    source = WORKSPACE_JS
    probe = source[ source.index("async function _workspaceProbeFile(") : ]
    probe = probe[: probe.index("\n}\n")]
    assert "_workspaceRouteForPath(path, 'raw')" in probe
    assert "'read'" not in probe
    # Only a 404 counts as absent; other failures are inconclusive and must
    # not be turned into a hard "missing" verdict.
    assert re.search(r"if\s*\(\s*status\s*===\s*404\s*\)\s*return false", probe)
    # A resolved response is not proof of existence: only an explicit 2xx is.
    assert re.search(r"status\s*>=\s*200", probe), "probe must require a 2xx status"
    assert re.search(r"status\s*<\s*300", probe), "probe must reject non-2xx statuses"
    # The body is dropped so a large file is not buffered for an existence check.
    assert "statusOnly:true" in probe


def test_api_contract_is_untouched():
    """This fix is frontend-only: no server behaviour or response shape changes.

    Keeping the API surface untouched is what lets the probe coexist with the
    pagination work for #6645 instead of competing with it.
    """
    routes = (ROOT / "api" / "routes.py").read_text(encoding="utf-8")
    handler = routes[ routes.index("def _handle_list_dir(") : ]
    handler = handler[: handler.index("\ndef ")]
    assert '"truncated"' not in handler
    assert "entries = list_dir(Path(workspace), rel_path)" in handler

    workspace_api = (ROOT / "api" / "workspace.py").read_text(encoding="utf-8")
    assert "MAX_LIST_ENTRIES" not in workspace_api
    assert "class _DirEntries" not in workspace_api


def test_workspace_js_stays_syntactically_valid():
    """A syntax error in workspace.js breaks the whole workspace panel."""
    if shutil.which("node") is None:
        pytest.skip("node is unavailable on this host")
    result = subprocess.run(
        ["node", "--check", str(ROOT / "static" / "workspace.js")],
        capture_output=True,
        text=True,
        env=_node_env(),
    )
    assert result.returncode == 0, result.stderr


# ── node driver: exercise the REAL browser-side decision logic ──────────────
# The static assertions above only prove the source TEXT mentions the probe.
# The driver below (tests/_workspace_path_exists_probe_driver.js) extracts the
# shipped functions out of static/workspace.js and runs them against a mocked
# api(), so the decision branches are verified as behaviour, not as a grep hit.


def test_browser_existence_logic_behaviour():
    """Drive the real _workspacePathExists / _workspaceProbeFile.

    Branches verified:
      1. the parent listing contains the name  -> true, no probe
      2. the listing is capped, the name is missing -> true via the direct probe
         (this is the bug: the file was present on disk but unreachable)
      3. the file genuinely does not exist      -> false (404 only)
      4. the listing request itself fails       -> still probes the file
      5. a non-404 probe failure               -> reported as not-openable,
         never as a fabricated "missing" verdict
      6. the probe resolves but carries NO status -> NOT reported as existing.
         A resolved response is not proof of existence; only an explicit 2xx
         is. This mirrors the upstream contract in
         tests/test_issue6867_artifacts_settled_recovery.py, where a missing
         artifact path must stay blocked.
    """
    if shutil.which("node") is None:
        pytest.skip("node is unavailable on this host")
    driver = ROOT / "tests" / "_workspace_path_exists_probe_driver.js"
    result = subprocess.run(
        ["node", str(driver), str(ROOT / "static" / "workspace.js")],
        capture_output=True,
        text=True,
        env=_node_env(),
    )
    # NOTE: a "not found" in stderr means the driver could not extract a
    # function from workspace.js — that is a REAL failure, not a missing node.
    # Skipping on it would hide the regression this test exists to catch.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL BRANCHES OK" in result.stdout
