"""Regression coverage for the #7170 gateway-worker profile leak.

On the gateway chat backend the detached worker thread used to call the
ambient ``get_config()``, which resolves through the *process-global* profile
(usually ``default``), not the profile that owns the session. The worker feeds
that config into ``_gateway_reasoning_effort_for_request()`` to read
``agent.reasoning_overrides`` and to coerce the effort against the model's
capabilities — so on a multi-profile instance a turn in profile B could pick up
profile A's per-model override (Codex reproduced it differentially: the session
profile resolved ``high`` while the gateway request carried ``none`` from the
other profile).

Fix: /api/chat/start now captures the session-owning profile's config snapshot
at dispatch time (``_gateway_session_owner_cfg`` — the same
``get_config_for_profile_home`` snapshot the in-process path uses, issue #3294)
and passes it into the gateway worker as ``session_cfg``. The worker uses that
for the override selection and the model-capability coercion instead of
resolving config on its own thread.

The regressions prove one profile's ``reasoning_overrides`` cannot affect
another profile's gateway request: with the ambient profile pinned to a config
that would yield ``none`` for the model, a session owned by profile B (whose
config yields ``high``) must still send ``reasoning_effort: high`` — and the
session-side and gateway-side resolvers must agree on the same dispatched
snapshot.
"""

from collections import OrderedDict
import json
from pathlib import Path

import pytest
import yaml

import api.gateway_chat as gateway_chat
import api.models as models
import api.routes as routes
import api.streaming as streaming
from api.config import STREAMS, create_stream_channel


_MODEL = "claude-opus-4-5"


def _write_cfg(home: Path, *, global_effort: str, override_effort: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    home.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "reasoning_effort": global_effort,
                    "reasoning_overrides": {_MODEL: override_effort},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Ambient profile A resolves ``none``; session profile B resolves ``high``.

    A = process-global profile the detached worker would fall back to.
    B = the profile that owns the session under test.
    """
    from api import config as cfg

    profile_a_home = tmp_path / "profiles" / "a"
    profile_b_home = tmp_path / "profiles" / "b"
    # Ambient (process-global) profile A: a per-model override pinning the
    # model to none, which must NOT leak into profile B's request.
    _write_cfg(profile_a_home, global_effort="max", override_effort="none")
    # Session profile B: same model explicitly pinned to high.
    _write_cfg(profile_b_home, global_effort="high", override_effort="high")

    # Pin the ambient resolver to profile A, simulating a worker thread that
    # fell back to the process-global profile.
    monkeypatch.setenv("HERMES_CONFIG_PATH", str(profile_a_home / "config.yaml"))
    cfg.reload_config()

    # Isolate the session-profile -> home resolution to profile B's config
    # without depending on the real on-disk profile directory layout. The
    # worker imports ``_get_profile_home`` from ``api.models`` at call time,
    # so patch it at its source.
    import api.models as _models_mod

    monkeypatch.setattr(
        _models_mod, "_get_profile_home", lambda profile: profile_b_home
    )

    yield cfg, profile_a_home, profile_b_home

    monkeypatch.delenv("HERMES_CONFIG_PATH", raising=False)
    cfg.reload_config()


def test_ambient_worker_config_reads_profile_a(two_profiles):
    """Baseline: the ambient resolver the buggy worker used sees profile A's
    ``none`` override — the leak source."""
    cfg, profile_a_home, profile_b_home = two_profiles
    ambient = cfg.get_config()
    # Profile A's explicit per-model override "none" is preserved as the
    # string "none" (the resolver keeps explicit "none", only omitting absent
    # or invalid effort) — this is the value the buggy worker would have sent.
    assert (
        gateway_chat._gateway_reasoning_effort_for_request(
            ambient, model=_MODEL, model_provider="anthropic"
        )
        == "none"
    )


def test_session_profile_config_reads_profile_b(two_profiles):
    """The fix target: resolving the session's own profile home reads profile
    B's ``high`` override even though the ambient resolver points at A."""
    cfg, profile_a_home, profile_b_home = two_profiles
    from api.config import get_config_for_profile_home

    session_cfg = get_config_for_profile_home(profile_b_home)
    assert (
        gateway_chat._gateway_reasoning_effort_for_request(
            session_cfg, model=_MODEL, model_provider="anthropic"
        )
        == "high"
    )


def test_gateway_worker_sends_session_profile_override_not_ambient(
    two_profiles, tmp_path, monkeypatch
):
    """End-to-end: a session owned by profile B must send B's ``high``, not the
    ambient profile A's ``none``."""
    cfg, profile_a_home, profile_b_home = two_profiles

    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", OrderedDict())

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
            yield b'data: [DONE]\n\n'

    def fake_urlopen(req, timeout=0):
        captured["body"] = req.data.decode("utf-8")
        return FakeResponse()

    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://gateway.local")
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_API_KEY", "secret-token")
    monkeypatch.setattr(gateway_chat.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        streaming, "_load_webui_prefill_context", lambda c: {"messages": []}
    )
    monkeypatch.setattr(
        streaming, "_prefill_messages_with_webui_context", lambda ctx, c: []
    )

    s = models.new_session(profile="b")
    s.pending_user_message = "Say hello"
    s.pending_attachments = []
    s.pending_started_at = 123
    s.save()
    stream_id = "stream-profile-isolation"
    s.active_stream_id = stream_id
    channel = create_stream_channel()
    STREAMS[stream_id] = channel

    gateway_chat._run_gateway_chat_streaming(
        s.session_id,
        "Say hello",
        _MODEL,
        str(tmp_path),
        stream_id,
        [],
        model_provider="anthropic",
    )

    payload = json.loads(captured["body"])
    # Profile B's per-model override wins; ambient profile A's "none" must not
    # leak into this request.
    assert payload["reasoning_effort"] == "high"


def test_dispatch_captures_session_profile_cfg_and_passes_to_gateway_worker(
    two_profiles, tmp_path, monkeypatch
):
    """Two-profile differential at the /api/chat/start dispatch boundary.

    Dispatching a session owned by profile B must capture B's config snapshot
    (via ``_gateway_session_owner_cfg``) and pass it to the gateway worker as
    ``session_cfg`` — with the ambient resolver still pinned to profile A.
    The same dispatched snapshot must resolve B's override through BOTH the
    session-side resolver (``configured_reasoning_effort_for_model``, the
    native/status decision point) and the gateway-side resolver
    (``_gateway_reasoning_effort_for_request``), while profile A's ``none``
    stays out. Reverting the fix (worker resolves config itself) yields
    ``session_cfg=None`` and this test fails.
    """
    cfg, profile_a_home, profile_b_home = two_profiles

    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(models, "SESSIONS", OrderedDict())

    captured_thread = {}

    class ImmediateThread:
        def __init__(self, *args, **kwargs):
            # ``kwargs`` here is the Thread() constructor payload
            # ({target, args, kwargs, daemon}); the real worker kwargs ride in
            # the inner ``kwargs["kwargs"]`` slot created by the Thread call.
            captured_thread["kwargs"] = kwargs
            self.args = args

        def start(self):
            return None

    monkeypatch.setattr(routes, "set_last_workspace", lambda workspace, **_kw: None)
    monkeypatch.setattr(
        routes, "create_stream_channel", lambda: create_stream_channel()
    )
    monkeypatch.setattr(routes.threading, "Thread", ImmediateThread)

    s = models.new_session(profile="b")
    s.pending_user_message = "Say hello"
    s.pending_attachments = []
    s.pending_started_at = 123
    s.title = "Profile B"
    s.messages = [{"role": "user", "content": "Say hello"}]
    s.save()

    response = routes._start_chat_stream_for_session(
        s,
        msg="Say hello",
        attachments=[],
        workspace=str(tmp_path),
        model=_MODEL,
        model_provider="anthropic",
        external_runtime_owned=True,
    )
    assert response and "stream_id" in response
    # Thread() constructor kwargs: {target, args, kwargs(worker), daemon}.
    # The gateway worker kwargs ride in the inner "kwargs" slot.
    thread_payload = captured_thread.get("kwargs") or {}
    worker_kwargs = thread_payload.get("kwargs") or {}
    # The gateway worker must have received the session-owner config snapshot.
    session_cfg = worker_kwargs.get("session_cfg")
    assert session_cfg is not None

    from api.config import configured_reasoning_effort_for_model

    # Session-side resolver and gateway-side resolver must agree on the same
    # dispatched snapshot — profile B's "high", never ambient A's "none".
    assert (
        configured_reasoning_effort_for_model(
            session_cfg, model_id=_MODEL, provider_id="anthropic"
        )
        == "high"
    )
    assert (
        gateway_chat._gateway_reasoning_effort_for_request(
            session_cfg, model=_MODEL, model_provider="anthropic"
        )
        == "high"
    )
    # Differential: the ambient (process-global) resolver still sees profile
    # A's "none" — if the dispatch had fallen back to it, this would leak.
    ambient_cfg = cfg.get_config()
    assert (
        gateway_chat._gateway_reasoning_effort_for_request(
            ambient_cfg, model=_MODEL, model_provider="anthropic"
        )
        == "none"
    )

    # Close the loop on the gateway side: feed the dispatch-captured
    # snapshot into the real worker and assert the outgoing request carries
    # profile B's "high" (not ambient A's "none").
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"done"}}]}\n\n'
            yield b'data: [DONE]\n\n'

    def fake_urlopen(req, timeout=0):
        captured["body"] = req.data.decode("utf-8")
        return FakeResponse()

    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_BASE_URL", "http://gateway.local")
    monkeypatch.setenv("HERMES_WEBUI_GATEWAY_API_KEY", "secret-token")
    monkeypatch.setattr(gateway_chat.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        streaming, "_load_webui_prefill_context", lambda c: {"messages": []}
    )
    monkeypatch.setattr(
        streaming, "_prefill_messages_with_webui_context", lambda ctx, c: []
    )

    stream_id = "stream-dispatch-captured-cfg"
    s.active_stream_id = stream_id
    channel = create_stream_channel()
    STREAMS[stream_id] = channel

    gateway_chat._run_gateway_chat_streaming(
        s.session_id,
        "Say hello",
        _MODEL,
        str(tmp_path),
        stream_id,
        [],
        model_provider="anthropic",
        session_cfg=session_cfg,
    )
    payload = json.loads(captured["body"])
    assert payload["reasoning_effort"] == "high"
