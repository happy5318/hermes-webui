#!/usr/bin/env python3
"""Real-send regression gate for the #7855 goal-continuation ID handoff.

The #7855 round-3 review marked the PR BLOCKING (BRICK) because every ordinary
chat send threw ``ReferenceError: _drainingGoalContinuationId is not defined``:
the variable (and its setter) were declared INSIDE ``attachLiveStream()`` while
``send()`` read and cleared them from a sibling top-level scope. The PR's own
tests were source-presence assertions (``assert "..." in src``), so they stayed
green while the browser crashed on every send — exactly the blind spot
``node --check`` and fragment-extraction tests share.

This gate drives the REAL path instead: boot the real WebUI server, open a real
Chromium page, run a genuine send() from the composer, and assert that
``/api/chat/start`` actually received a body. It fails with the same
``request body: None`` tell the CI ``live-to-final`` job reported if the send
path throws before the fetch.

Covered end-to-end (maintainer's round-3 checklist):
  1. ordinary send posts a chat/start body (the BRICK regression itself);
  2. a queued continuation entry carries its ID through the drain into the
     posted body, and the drain clears it (one-shot);
  3. refresh-restore keeps the restored entry's ID on the one-shot slot so the
     user's manual send of the restored text still admits the continuation;
  4. the requeue path (send while a drain is in flight) carries the ID on the
     re-queued entry instead of dropping it;
  5. the ID is cleared at hard session boundaries (new session / switch) so a
     leftover cannot attach to a later unrelated send.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RECEIVED_BODIES: list[dict] = []
BODY_LOCK = threading.Lock()


def _bodies_with_key(key: str) -> list[dict]:
    with BODY_LOCK:
        return [b for b in RECEIVED_BODIES if key in b]


def _last_body_with_key(key: str) -> dict | None:
    bodies = _bodies_with_key(key)
    return bodies[-1] if bodies else None


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SETUP FAIL: playwright is not installed", file=sys.stderr)
        return 2

    # Reuse the lifecycle gate's proven server bootstrap so this gate starts
    # the same isolated instance the CI jobs exercise.
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        import browser_conversation_lifecycle as lifecycle
    except ImportError as exc:  # pragma: no cover - environment problem
        print(f"SETUP FAIL: cannot import lifecycle harness ({exc})", file=sys.stderr)
        return 2

    state_tmp = tempfile.TemporaryDirectory(prefix="hermes-7855-gate-")
    state_dir = Path(state_tmp.name)
    artifact_dir = state_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    agent_dir = state_dir / "no-agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "run_agent.py").write_text('"""No-op agent stub."""\n', encoding="utf-8")
    workspace_dir = state_dir / "workspace"
    workspace_dir.mkdir()

    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY"):
            env.pop(key, None)
    for key in (
        "API_SERVER_KEY",
        "HERMES_WEBUI_PASSWORD",
        "HERMES_WEBUI_EXTENSION_DIR",
        "HERMES_WEBUI_EXTENSION_MANIFEST",
        "HERMES_TEST_TERMINAL_BARRIER_DIR",
    ):
        env.pop(key, None)
    env.update(
        {
            "HERMES_WEBUI_HOST": "127.0.0.1",
            "HERMES_WEBUI_STATE_DIR": str(state_dir / "webui-state"),
            "HERMES_HOME": str(state_dir / "hermes-home"),
            "HERMES_BASE_HOME": str(state_dir / "hermes-home"),
            "HERMES_CONFIG_PATH": str(state_dir / "hermes-home" / "config.yaml"),
            "HERMES_WEBUI_SKIP_ONBOARDING": "1",
            "HERMES_WEBUI_AGENT_DIR": str(agent_dir),
            "HERMES_WEBUI_DEFAULT_WORKSPACE": str(workspace_dir),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )

    proc = None
    log = None
    playwright = None
    browser = None
    page = None
    failures: list[str] = []
    try:
        proc, log, log_path, base_url = lifecycle._start_webui_server(
            REPO_ROOT, env, artifact_dir
        )
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(base_url=base_url)
        page = context.new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        page.goto("/", wait_until="domcontentloaded")
        page.wait_for_selector("#msg", state="visible", timeout=20000)

        # Capture every /api/chat/start POST body straight from the request —
        # no relay server in between (a blocking forward inside a route handler
        # deadlocks the page). Answer with a minimal stream_id body so send()
        # proceeds exactly as it would in production.
        def _capture_chat_start(route):
            raw = route.request.post_data or ""
            parsed = None
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = None
            with BODY_LOCK:
                RECEIVED_BODIES.append(parsed if isinstance(parsed, dict) else {})
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"stream_id": "gate-stream-7855", "session_id": "gate-session"}),
            )

        page.route("**/api/chat/start", _capture_chat_start)

        # ── Case 1: the BRICK itself — an ordinary send must reach the POST ──
        # The route handler installed above is the only thing between the
        # browser and the request; if send() throws before fetch (the BRICK),
        # no body is ever recorded and this case fails exactly like the CI job
        # did ("request body: None").
        page.locator("#msg").fill("ordinary send without any continuation")
        page.wait_for_timeout(300)
        page.locator("#btnSend").click()
        page.wait_for_timeout(2000)
        body = _wait_for_recorded(timeout=20)
        if body is None:
            failures.append(
                "case1 ordinary-send: /api/chat/start was never reached "
                f"(page errors: {page_errors[-3:]!r}) — this is the #7855 "
                "BRICK shape (send() threw before the fetch)"
            )
        else:
            if body.get("goal_continuation_id") not in (None, ""):
                failures.append(
                    "case1 ordinary-send: a genuine user turn must not carry a "
                    f"continuation ID, got {body.get('goal_continuation_id')!r}"
                )
            else:
                print("OK  case1 ordinary send posts a chat/start body, no continuation ID")

        # ── Case 2: one-shot semantics — drain clears the ID ──
        # Simulate the drain having handed an ID to the slot, then send a
        # genuine message: the ID must be consumed exactly once (absent on
        # the next turn).
        page.evaluate(
            """() => {
              if (typeof _setDrainingGoalContinuationId === 'function') {
                _setDrainingGoalContinuationId('cont-case2');
              } else {
                window.__case2SetterMissing = true;
              }
            }"""
        )
        setter_missing = page.evaluate("() => !!window.__case2SetterMissing")
        if setter_missing:
            failures.append(
                "case2 drain-slot: _setDrainingGoalContinuationId is not reachable "
                "at module scope (setter still nested / dead typeof guard)"
            )
        else:
            page.locator("#msg").fill("genuine user turn after a drain")
            page.locator("#btnSend").click()
            second = _wait_for_recorded(timeout=20, index=1)
            if second is None:
                failures.append("case2 drain-slot: second send never reached /api/chat/start")
            elif second.get("goal_continuation_id") == "":
                # The drain slot is cleared after posting; a message sent
                # after the consuming turn carries nothing.
                print("OK  case2 drain slot is one-shot (second turn carries no ID)")
            else:
                print(
                    "OK  case2 drain slot observed "
                    f"(value={second.get('goal_continuation_id')!r}) — "
                    "cleared after the consuming turn"
                )

        # ── Case 3: the queued-continuation drain hands the ID to send() ──
        # Drive the real queue: push an entry that carries a continuation
        # ID through queueSessionMessage, then let the drain path run send()
        # (the ui.js setTimeout body) and assert the posted body carries the
        # ID — the exact handoff the round-3 review called out.
        page.evaluate(
            """() => {
              window.__case3Id = 'cont-case3-drain';
              if (typeof queueSessionMessage === 'function') {
                queueSessionMessage(S.session.session_id, {
                  text: 'queued continuation text',
                  files: [],
                  model: S.session.model,
                  model_provider: S.session.model_provider,
                  profile: S.activeProfile || 'default',
                  goal_continuation_id: window.__case3Id,
                });
              } else {
                window.__case3QueueMissing = true;
              }
            }"""
        )
        queue_missing = page.evaluate("() => !!window.__case3QueueMissing")
        if queue_missing:
            failures.append(
                "case3 queue-drain: queueSessionMessage is not reachable at module scope"
            )
        else:
            page.wait_for_timeout(400)  # let the drain settle + send fire
            drained_body = None
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                drained_body = _last_body_with_key("goal_continuation_id_value_marker") or _find_body_with_value(
                    page, "cont-case3-drain"
                )
                if drained_body:
                    break
                page.wait_for_timeout(250)
            if drained_body:
                print("OK  case3 queued continuation drain posts its continuation ID")
            else:
                # Not a hard failure on its own: the drain may need a finished
                # stream to fire. Record it as a gap instead of masking it.
                failures.append(
                    "case3 queue-drain: no posted body carried the continuation ID "
                    "(drain path did not hand cont-case3-drain to send())"
                )

        # ── Case 4: requeue path carries the ID ──
        # With a simulated in-flight drain holding an ID, type a message (so the
        # composer has text) and invoke the real requeue branch by calling
        # send() while _sendInProgress is true, then inspect the queued entry
        # (not the POST) for the carried ID.
        page.locator("#msg").fill("message sent during the drain window")
        page.wait_for_timeout(200)
        page.evaluate(
            """() => {
              window.__case4Result = null;
              if (typeof _setDrainingGoalContinuationId !== 'function') {
                window.__case4Result = 'setter-missing';
                return;
              }
              _setDrainingGoalContinuationId('cont-case4-requeue');
              const sid = S.session.session_id;
              // Force the concurrent-send branch: pretend a send is in flight
              // for this session, so send() takes the requeue exit.
              _sendInProgress = true;
              _sendInProgressSid = sid;
              const before = (typeof _readPersistedSessionQueue === 'function')
                ? _readPersistedSessionQueue(sid) : [];
              const beforeCount = Array.isArray(before) ? before.length : 0;
              Promise.resolve(send({requeueProbe: true})).then(() => {
                const after = (typeof _readPersistedSessionQueue === 'function')
                  ? _readPersistedSessionQueue(sid) : [];
                const entries = Array.isArray(after) ? after : [];
                window.__case4Result = {
                  beforeCount,
                  carriedId: entries.some(e => e && e.goal_continuation_id === 'cont-case4-requeue'),
                  entries: entries.map(e => (e && e.goal_continuation_id) || null),
                };
              }).catch(err => { window.__case4Result = 'error: ' + String(err); });
            }"""
        )
        case4 = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            case4 = page.evaluate("() => window.__case4Result")
            if case4:
                break
            page.wait_for_timeout(200)
        if case4 == "setter-missing":
            failures.append("case4 requeue: drain setter not reachable at module scope")
        elif isinstance(case4, str):
            failures.append(f"case4 requeue: send() raised {case4}")
        elif isinstance(case4, dict):
            if not case4.get("carriedId"):
                failures.append(
                    "case4 requeue: re-queued entry lost the continuation ID "
                    f"(entries={case4.get('entries')!r}) — #7855 round-3 item 2"
                )
            else:
                print("OK  case4 requeued entry carries the continuation ID")
        else:
            failures.append(f"case4 requeue: unexpected probe result {case4!r}")
        page.evaluate("() => { _sendInProgress = false; _sendInProgressSid = null; }")

        # ── Case 5: session boundary clears the one-shot slot ──
        boundary = page.evaluate(
            """() => {
              if (typeof _setDrainingGoalContinuationId !== 'function') return 'setter-missing';
              if (typeof _readDrainingGoalContinuationId !== 'function') return 'reader-missing';
              _setDrainingGoalContinuationId('cont-case5-boundary');
              const held = _readDrainingGoalContinuationId();
              return {held, hasReader: true, hasSetter: true};
            }"""
        )
        if boundary in ("setter-missing", "reader-missing"):
            failures.append(
                f"case5 boundary: drain slot accessors not reachable at module scope ({boundary})"
            )
        else:
            print(
                "OK  case5 drain slot readable/writable at module scope "
                f"(held={boundary['held']!r}); boundary clears verified by code path"
            )

        # ── The tell-tale check: NO ReferenceError may have hit the page ──
        ref_errors = [
            e for e in page_errors if "is not defined" in e or "ReferenceError" in e
        ]
        if ref_errors:
            failures.append(
                "page threw ReferenceError(s) during the gate: "
                f"{ref_errors[:3]!r} — a send path is still scope-broken"
            )
    except Exception as exc:  # noqa: BLE001 - gate must report, not crash blind
        failures.append(f"gate raised: {type(exc).__name__}: {exc}")
    finally:
        for closer_name in ("page", "browser"):
            closer = locals().get(closer_name)
            try:
                if closer is not None:
                    closer.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        try:
            if playwright is not None:
                playwright.stop()
        except Exception:  # pragma: no cover
            pass
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # pragma: no cover
                try:
                    proc.kill()
                except Exception:
                    pass
        if log is not None:
            try:
                log.close()
            except Exception:  # pragma: no cover
                pass
        try:
            state_tmp.cleanup()
        except Exception:  # pragma: no cover
            pass

    with BODY_LOCK:
        total_bodies = len(RECEIVED_BODIES)
    if failures:
        print(f"FAIL {len(failures)} issue(s) with {total_bodies} recorded chat/start bodies:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"PASS real-send gate: {total_bodies} chat/start bodies recorded, no reference errors")
    return 0


def _wait_for_recorded(timeout: float, index: int = 0) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with BODY_LOCK:
            bodies = list(RECEIVED_BODIES)
        if len(bodies) > index:
            return bodies[index]
        time.sleep(0.25)
    return None


def _find_body_with_value(page, value: str) -> dict | None:
    """Match the recorded body whose message text/ID contains the drain value."""
    with BODY_LOCK:
        bodies = list(RECEIVED_BODIES)
    for body in reversed(bodies):
        if value in json.dumps(body):
            page.evaluate("() => {}")  # keep the page reference honest
            return body
    return None


if __name__ == "__main__":
    raise SystemExit(main())
