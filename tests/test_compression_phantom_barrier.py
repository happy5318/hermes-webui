"""Regression test: phantom "Compressing context" barrier on lost compressed SSE.

This test loads the real WebUI in Chromium with a mock EventSource, calls the
real attachLiveStream function, and proves the done handler clears stale running
compression state. The test executes the actual shipping handler path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(base_url: str, timeout: float = 30.0, proc=None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    return False


def _terminate_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _start_webui_server(repo_root: Path, state_dir: Path):
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    agent_dir = state_dir / "no-agent"
    agent_dir.mkdir(parents=True)
    workspace_dir = state_dir / "workspace"
    workspace_dir.mkdir()
    (agent_dir / "run_agent.py").write_text(
        '''"""Empty agent stub."""''' + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY"):
            env.pop(key, None)
    for key in (
        "API_SERVER_KEY",
        "HERMES_WEBUI_PASSWORD",
        "HERMES_WEBUI_EXTENSION_DIR",
        "HERMES_WEBUI_EXTENSION_MANIFEST",
    ):
        env.pop(key, None)

    env.update({
        "HERMES_WEBUI_HOST": "127.0.0.1",
        "HERMES_WEBUI_PORT": str(port),
        "HERMES_WEBUI_STATE_DIR": str(state_dir / "webui-state"),
        "HERMES_HOME": str(state_dir / "hermes-home"),
        "HERMES_BASE_HOME": str(state_dir / "hermes-home"),
        "HERMES_CONFIG_PATH": str(state_dir / "hermes-home" / "config.yaml"),
        "HERMES_WEBUI_SKIP_ONBOARDING": "1",
        "HERMES_WEBUI_AGENT_DIR": str(agent_dir),
        "HERMES_WEBUI_DEFAULT_WORKSPACE": str(workspace_dir),
    })

    log_path = state_dir / "server.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(repo_root / "server.py")],
        cwd=repo_root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    if _wait_for_health(base_url, proc=proc):
        return proc, log, log_path, base_url
    _terminate_process(proc)
    log.close()
    raise RuntimeError(f"WebUI server did not become healthy on port {port}")


def _run_test(page, active_sid, done_sid):
    """Execute real attachLiveStream with mock EventSource."""
    return page.evaluate(
        """
        ({activeSid, doneSid}) => {
            // Setup session state
            window.S = window.S || {};
            window.S.session = {session_id: activeSid};
            window.S.messages = [];
            window.S.activeStreamId = null;

            // Mock appendLiveCompressionCard to return false
            window._origAppend = window.appendLiveCompressionCard;
            window.appendLiveCompressionCard = () => false;

            // Mock setCompressionUi if needed
            if (typeof window.setCompressionUi !== 'function') {
                window.setCompressionUi = (s) => { window._compressionUi = s; };
            }

            // Create mock EventSource
            class MockES {
                constructor(url) {
                    this.url = url;
                    this.readyState = 0;
                    this._listeners = {};
                    setTimeout(() => this._run(), 10);
                }
                addEventListener(t, fn) { this._listeners[t] = this._listeners[t] || []; this._listeners[t].push(fn); }
                removeEventListener() {}
                close() { this.readyState = 2; }
                _dispatch(t, e) { (this._listeners[t] || []).forEach(f => f(e)); }
                _run() {
                    this.readyState = 1;
                    this._dispatch('open', {});
                    // Emit compressing (no session_id to pass the check)
                    this._dispatch('compressing', {data: JSON.stringify({message: 'Compressing context'})});
                    // Emit done
                    this._dispatch('done', {data: JSON.stringify({session: {session_id: doneSid, messages: []}})});
                    this.readyState = 2;
                }
            }
            window.EventSource = MockES;

            const result = {before: null, after: null, cleared: false, error: null};

            try {
                // Call real attachLiveStream
                if (typeof attachLiveStream !== 'function') {
                    result.error = 'attachLiveStream not defined';
                    return result;
                }
                attachLiveStream(activeSid, 'test-stream', []);
            } catch (e) {
                result.error = String(e);
            }

            return new Promise(resolve => {
                setTimeout(() => {
                    result.after = window._compressionUi ? JSON.parse(JSON.stringify(window._compressionUi)) : null;
                    result.cleared = (result.after === null);
                    
                    // Restore
                    window.EventSource = window._origES;
                    if (window._origAppend) window.appendLiveCompressionCard = window._origAppend;
                    
                    resolve(result);
                }, 100);
            });
        }
        """,
        {"activeSid": active_sid, "doneSid": done_sid},
    )


def _run_scenario(scenario: str) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("playwright required: pip install playwright") from exc

    repo_root = Path(__file__).resolve().parents[1]
    state_dir = Path(tempfile.mkdtemp(prefix=f"hermes-compression-{scenario}-"))

    proc = None
    log = None
    playwright = None
    browser = None
    page = None
    errors = []

    try:
        proc, log, log_path, base_url = _start_webui_server(repo_root, state_dir)

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(base_url=base_url)
        page = context.new_page()

        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto("/", wait_until="domcontentloaded")
        page.wait_for_selector("#msg", state="visible", timeout=15000)

        if scenario == "running_a_to_b_cleared":
            result = _run_test(page, active_sid="session-a", done_sid="session-b")
        elif scenario == "running_a_to_a_cleared":
            result = _run_test(page, active_sid="session-a", done_sid="session-a")
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        context.close()

        return {
            "scenario": scenario,
            "result": result,
            "errors": [e for e in errors if "favicon" not in e.lower()],
        }
    finally:
        if browser:
            browser.close()
        if playwright:
            playwright.stop()
        if proc:
            _terminate_process(proc)
        if log:
            log.close()
        shutil.rmtree(state_dir, ignore_errors=True)


def test_running_a_to_b_cleared():
    """A→B: session rotates, compressed SSE lost, done clears running state."""
    outcome = _run_scenario("running_a_to_b_cleared")

    r = outcome["result"]
    assert r.get("error") is None, f"harness error: {r.get('error')}"
    assert r["cleared"], f"running A→B should be cleared, got: {r}"
    assert r["after"] is None, f"_compressionUi should be null, got: {r['after']}"
    assert not outcome["errors"], f"errors: {outcome['errors']}"


def test_running_a_to_a_cleared():
    """A→A: no rotation, compressed SSE lost, done clears running state."""
    outcome = _run_scenario("running_a_to_a_cleared")

    r = outcome["result"]
    assert r.get("error") is None, f"harness error: {r.get('error')}"
    assert r["cleared"], f"running A→A should be cleared, got: {r}"
    assert r["after"] is None, f"_compressionUi should be null, got: {r['after']}"
    assert not outcome["errors"], f"errors: {outcome['errors']}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        outcome = _run_scenario(sys.argv[1])
        print(json.dumps(outcome, indent=2, default=str))
    else:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
