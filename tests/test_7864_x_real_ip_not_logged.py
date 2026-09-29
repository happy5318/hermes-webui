"""#7864: ``X-Real-IP`` alone must never populate the request log's
``forwarded_for``.

nginx relays client-supplied request headers through by default
(``proxy_pass_request_headers on``) rather than overwriting ``X-Real-IP``, so a
header that a client controls cannot be allowed to name the log's client IP:
doing so would re-open the #7863 spoof through a different header, and a
fail2ban jail keyed on ``forwarded_for`` would then ban an arbitrary address.

The pre-#7864 logger never read `X-Real-IP`. Keep it that way: the request-log
resolver consumes only `X-Forwarded-For`, walks it right-to-left from the
right-most untrusted hop, and otherwise resolves to the raw socket peer.
"""

from tests.test_security_review_fixes import _Handler


def _trusted_env(monkeypatch):
    from api import routes

    monkeypatch.setenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", "1")
    monkeypatch.setenv("HERMES_WEBUI_TRUSTED_PROXY_CIDRS", "10.9.9.0/24")
    # _trusted_proxy_networks() caches per-process; clear it via the repo idiom.
    clear = getattr(routes._trusted_proxy_networks, "cache_clear", None)
    if clear is not None:
        clear()
    return routes


def test_x_real_ip_alone_never_feeds_forwarded_for(monkeypatch):
    """A loopback/trusted peer sending only `X-Real-IP` resolves to the peer."""
    routes = _trusted_env(monkeypatch)

    handler = _Handler(
        client_ip="10.9.9.7",  # trusted proxy (non-loopback, in the CIDR)
        headers={"X-Real-IP": "203.0.113.99"},
    )

    resolved = routes._forwarded_client_ip_from_trusted_proxy(handler)

    # The raw peer speaks for itself; the client-supplied X-Real-IP is ignored.
    assert resolved == "10.9.9.7"


def test_x_real_ip_does_not_override_the_xff_chain(monkeypatch):
    """`X-Real-IP` cannot replace a real client hop in an `X-Forwarded-For`."""
    routes = _trusted_env(monkeypatch)

    handler = _Handler(
        client_ip="10.9.9.7",
        headers={
            "X-Forwarded-For": "198.51.100.23",
            "X-Real-IP": "203.0.113.99",
        },
    )

    resolved = routes._forwarded_client_ip_from_trusted_proxy(handler)

    assert resolved == "198.51.100.23"


def test_x_real_ip_alone_from_a_trusted_peer_absent_from_log_fields(monkeypatch):
    """End-to-end: the logged field falls back to the peer, never the header."""
    routes = _trusted_env(monkeypatch)

    handler = _Handler(
        client_ip="10.9.9.7",
        headers={"X-Real-IP": "203.0.113.99"},
    )

    # Same gate the request-log boundary uses: trusted-peer check resolves the
    # value that lands in `forwarded_for`.
    assert routes._raw_peer_is_trusted_proxy(handler)
    resolved = routes._forwarded_client_ip_from_trusted_proxy(handler)
    assert resolved != "203.0.113.99"
