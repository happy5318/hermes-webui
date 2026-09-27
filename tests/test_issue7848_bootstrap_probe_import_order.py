"""The bootstrap probe must survive an interpreter that relaunches mid-script.

``_python_can_run_webui_and_agent`` asks a candidate interpreter to import the
WebUI dependencies *and* the Hermes Agent in one ``-c`` snippet. On PM-managed
installs the agent's own ``prepare_launch`` re-execs that snippet
(``hermes_cli/venv_sync.py``) with ``-I`` — which implies ``-E -s`` and drops
the caller's ``PYTHONPATH`` — and then replays the snippet *from the top*. In
that replay the agent import is what activates the managed runtime's dependency
path, so it must come first: any statement before it is evaluated in an
environment that has not been prepared yet.

With ``import yaml`` first the probe died with
``ModuleNotFoundError: No module named 'yaml'`` on the relaunched interpreter,
bootstrap concluded that *no* interpreter could serve both, and on the resulting
systemd restart loop the WebUI stayed down indefinitely (#7848). The error was
reported by the relaunched process, not by the interpreter the probe started
with, which is why the traceback location is the tell.
"""
from __future__ import annotations

import re
from pathlib import Path

import bootstrap


def _probe_source() -> str:
    """The probe body bootstrap hands to the candidate interpreter."""
    source = Path(bootstrap.__file__).read_text(encoding="utf-8")
    match = re.search(r"script\s*=\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')", source)
    assert match, "could not find the probe script literal in bootstrap.py"
    return eval(match.group(1))  # noqa: S307 - literal from our own source tree


def test_probe_imports_the_agent_before_any_webui_dependency():
    """The agent import must be the statement that runs first.

    Ordering is the whole fix: in a relaunch, importing the agent is what adds
    the managed runtime's dependency path, so a WebUI dependency imported before
    it is evaluated against an unprepared environment and the probe fails with a
    misleading ``No module named ...``.
    """
    statements = [
        line.strip()
        for line in _probe_source().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    agent_index = next(
        (i for i, s in enumerate(statements) if "run_agent" in s), None
    )
    assert agent_index is not None, "the probe must import the agent"

    preceding = statements[:agent_index]
    assert not preceding, (
        "statements run before the agent import are evaluated before the "
        "managed runtime activates its dependency path: "
        f"{preceding!r}"
    )


def test_probe_still_covers_the_webui_dependency():
    """Guard against "fixing" this by dropping the dependency import entirely."""
    assert "import yaml" in _probe_source()
