"""The large-string probe fast path must never trade correctness for speed.

``_redact_fn_cached`` re-runs the full redactor for every string above
``_REDACT_CACHE_MAX_TEXT_LEN`` (fail-closed against runtime-registered
patterns, see ``test_redact_large_string_cache.py``). On real sessions those
strings are long reasoning/tool blobs that dominate a session GET, so a
narrow probe now skips the redactor when the text contains none of the
credential SHAPES the built-in redactor can rewrite.

Contracts pinned here:
  1. probe-hit large strings redact exactly like the uncached redactor;
  2. probe-clean large strings come back byte-identical (the probe may only
     skip passes that provably cannot change the text);
  3. a runtime-registered pattern (unknown to the static probe) is still
     redacted afterwards — the fast path disables itself fail-closed;
  4. every built-in credential shape trips the probe (no probe miss).
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import helpers  # noqa: E402
from api.helpers import (  # noqa: E402
    _LARGE_STRING_PROBE_RE,
    _REDACT_CACHE_MAX_TEXT_LEN,
)


def _fast_path_enabled():
    # Read through the module: sibling tests reload api.helpers with a stubbed
    # agent.redact, and a from-import binding would keep pointing at the
    # pre-reload functions (which read the pre-reload globals).
    return helpers._probe_fast_path_enabled()


def _redact_cached(text):
    return helpers._redact_fn_cached(text)


def _redact_uncached(text):
    return helpers._redact_fn_uncached(text)


def _rearm_fast_path():
    """Reset plugin patterns and re-arm the latch against the live registry.

    Fail-closed is monotonic in production; tests that need the default
    baseline call this after any sibling reset/reload.
    """
    redact = sys.modules.get("agent.redact")
    reset = getattr(redact, "_reset_plugin_redaction_patterns", None)
    if reset is not None:
        reset()
    full = getattr(helpers._redact_fn_uncached, "__name__", "") == "_combined_redact"
    helpers._BUILTIN_PREFIX_COUNT = helpers._builtin_prefix_substring_count()
    helpers._probe_fast_path_allowed = full and helpers._BUILTIN_PREFIX_COUNT > 0


@pytest.fixture(autouse=True)
def _default_fast_path():
    """Run each test against a clean, freshly re-armed fast path.

    Sibling redaction tests register runtime patterns (or reload api.helpers
    with a stubbed agent.redact) in the shared process; both leave the latch
    tripped, because fail-closed is monotonic. Re-arm it here from the live
    registry so this module's probe assertions see the intended baseline, and
    drop any plugin patterns the siblings left behind.
    """
    _rearm_fast_path()
    baseline = helpers._probe_fast_path_allowed
    yield
    _rearm_fast_path()
    helpers._probe_fast_path_allowed = baseline


def _big(size: int, secret: str) -> str:
    unit = "lorem ipsum dolor sit amet "
    text = (unit * (size // len(unit) + 1) + secret + unit * 64)
    assert len(text) > _REDACT_CACHE_MAX_TEXT_LEN
    return text


# ── 1. probe hits still redact ───────────────────────────────────────────

@pytest.mark.parametrize("secret", [
    "sk-" + "a1b2c3d4e5f6" * 3,                       # OpenRouter-style
    "AKIA" + "ABCDEFGHIJKLMNOP",                      # AWS access key id
    "ghp_" + "A1b2C3d4E5f6G7h8" * 3,                  # GitHub PAT
    "AIza" + "aB3dE6gH9jK2mN5pQ8rS1tU4vW7xY0z",       # Google API key
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc",  # JWT
    "sk-abc\x1bdef1234567890abcdefghij",              # control-split token
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEabc\n",     # PEM header
    "123456789:AAHdEfGhIjKlMnOpQrStUvWxYz012345678",  # Telegram bot token
    "postgres://user:password@host:5432/db",          # URL userinfo
    'my_api_key="sk-supersecretvalue123456"',         # assignment form
])
def test_probe_hit_large_string_redacts(secret):
    text = _big(_REDACT_CACHE_MAX_TEXT_LEN + 4096, secret)
    assert _LARGE_STRING_PROBE_RE.search(text), "test secret must trip the probe"
    out = _redact_cached(text)
    oracle = _redact_uncached(text)
    # The fast path must reproduce the redactor exactly, whatever the redactor
    # does with the blob (mask, rewrite, or pass it through).
    assert out == oracle
    if oracle != text:
        # The redactor rewrites this shape, so the credential material itself
        # must not survive into the fast-path output.
        assert secret not in out, "fast path leaked material the redactor removed"


# ── 2. probe-clean large strings pass through byte-identical ──────────────

def test_probe_clean_large_string_passes_through():
    """Plain prose (reasoning blobs, tool output without credentials)."""
    text = "the tool returned lorem ipsum output with no credentials at all " * 800
    assert len(text) > _REDACT_CACHE_MAX_TEXT_LEN
    assert not _LARGE_STRING_PROBE_RE.search(text)
    assert _redact_cached(text) == text


def test_probe_clean_output_equals_full_redactor():
    """Identity is only safe because the redactor cannot change the text —
    prove that on a real payload-shaped blob."""
    text = ("thinking about the refactor: the fast path must stay honest " * 400)
    assert not _LARGE_STRING_PROBE_RE.search(text)
    assert _redact_uncached(text) == text


# ── 3. runtime-registered pattern disables the fast path (fail-closed) ────

def test_runtime_pattern_disables_fast_path(monkeypatch):
    # Pin the REAL agent.redact BEFORE api.helpers (re)builds its redactor:
    # sibling tests leave a fake redact_sensitive_text captured inside a
    # previously built _combined_redact closure, which would make the "real
    # registry masks the token" assertion meaningless.
    redact = pytest.importorskip("agent.redact")
    if not hasattr(redact, "register_redaction_patterns"):
        pytest.skip("installed agent.redact has no runtime pattern registry")

    reset = getattr(redact, "_reset_plugin_redaction_patterns", None)
    if reset is None:
        pytest.skip("installed agent.redact has no registry reset seam")

    monkeypatch.setitem(sys.modules, "agent", importlib.import_module("agent"))
    monkeypatch.setitem(sys.modules, "agent.redact", redact)
    helpers_reloaded = importlib.reload(helpers)

    fn = helpers_reloaded._redact_fn_uncached
    if getattr(fn, "__name__", "") != "_combined_redact":
        pytest.skip("api.helpers was built without the agent redactor")
    captured = [c.cell_contents for c in (fn.__closure__ or ())]
    if not any(getattr(c, "__module__", "").startswith("agent.redact") or getattr(c, "__name__", "") == "redact_sensitive_text" for c in captured):
        pytest.skip("helper redactor did not bind agent.redact's function in this build")

    reset()
    _rearm_fast_path()

    token = "nvapi-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2W3x4Y5z6A7b8C9d0"
    text = ("see https://example.invalid/docs lorem ipsum " * 500) + token + " tail"
    assert len(text) > _REDACT_CACHE_MAX_TEXT_LEN
    # The shape is unknown to the static probe on purpose (registration is the
    # only thing that teaches the redactor about it).
    assert not _LARGE_STRING_PROBE_RE.search(text)

    # Fast path is active BEFORE registration: clean blob, no rewrite.
    assert _fast_path_enabled() is True
    assert _redact_cached(text) == text

    accepted = redact.register_redaction_patterns([r"nvapi-[A-Za-z0-9]{60}"], source="test-probe-fail-closed")
    assert accepted == 1
    try:
        # Registration must flip the fast path off, so the secret is masked.
        assert _fast_path_enabled() is False
        out = _redact_cached(text)
        assert token not in out, "fail-closed latch opened but the secret was not masked"
        assert out == _redact_uncached(text)
    finally:
        reset()


# ── 4. every built-in credential shape trips the probe ───────────────────

def test_stub_build_disables_fast_path(monkeypatch):
    """No countable agent registry → the probe fast path must stay OFF.

    Mirrors the suite's "hermes-agent not found" mode: ``_build_redact_fn``
    binds the fallback redactor when the agent one is missing, and the probe
    must not signal "safe to skip".

    Deliberately does NOT reload api.helpers: a reload rebuilds the combined
    redactor's closure around whatever ``agent.redact`` is in ``sys.modules``
    at that instant, and restoring ``sys.modules`` afterwards does not rebuild
    it — sibling tests would keep the fake forever. Snapshot the few module
    attributes instead; monkeypatch restores them on teardown.
    """
    snapshot = {
        key: getattr(helpers, key)
        for key in ("_probe_fast_path_allowed", "_BUILTIN_PREFIX_COUNT")
    }
    big = "plain lorem ipsum reasoning blob with no credentials at all " * 800
    assert len(big) > _REDACT_CACHE_MAX_TEXT_LEN
    assert not _LARGE_STRING_PROBE_RE.search(big)

    # Simulate the fallback build: the registry reports zero usable prefixes,
    # so the "same count as import time" check latches the gate off.
    monkeypatch.setattr(helpers, "_builtin_prefix_substring_count", lambda: 0)
    helpers._probe_fast_path_allowed = True
    helpers._BUILTIN_PREFIX_COUNT = 56

    assert helpers._builtin_prefix_substring_count() == 0
    assert helpers._probe_fast_path_enabled() is False
    # Fail-closed: even a probe-clean blob re-runs the redactor.
    assert helpers._redact_fn_cached(big) == helpers._redact_fn_uncached(big)
    # monkeypatch undoes the two setattrs; the latch is monotonic in
    # production, so leave it latched — the next _rearm_fast_path() re-arms it.
    helpers._probe_fast_path_allowed = snapshot["_probe_fast_path_allowed"]

def test_every_credential_shape_trips_probe():
    from agent.redact import redact_sensitive_text as agent_redact

    samples = [
        "sk-" + "a1b2c3d4e5f6" * 3,
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "A1b2C3d4E5f6G7h8" * 3,
        "AIza" + "aB3dE6gH9jK2mN5pQ8rS1tU4vW7xY0z",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc",
        "sk-abc\x1bdef1234567890abcdefghij",
        "sk-abc\u200bdef1234567890abcdefghij",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEabc\n",
        "123456789:AAHdEfGhIjKlMnOpQrStUvWxYz012345678",
        "postgres://user:password@host:5432/db",
        'my_api_key="sk-supersecretvalue123456"',
    ]
    misses = []
    for sample in samples:
        changed = agent_redact(sample, force=True) != sample
        probed = bool(_LARGE_STRING_PROBE_RE.search(sample))
        if changed and not probed:
            misses.append(sample[:40])
    assert not misses, f"probe misses agent-redactable shapes: {misses}"


def test_probe_ignores_clean_text():
    clean = [
        "plain reasoning about refactors and tests " * 50,
        "QmFzZTY0IGRlY29kZWQgYmxvYiB3aXRoIG5vIGNyZWRlbnRpYWxzIGF0IGFsbA==",
        "quoting a vendor prefix like sk- in prose without a real token body",
    ]
    for text in clean:
        assert not _LARGE_STRING_PROBE_RE.search(text), f"probe over-triggers on {text[:40]!r}"
