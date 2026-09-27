import json
import io

import pytest

from server import Handler


@pytest.fixture
def log_output(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr("api.request_logging._STREAM", output)
    return output


def test_log_request_handles_malformed_request_without_path(log_output):
    """Malformed request lines can call log_request before path is assigned."""
    handler = Handler.__new__(Handler)
    handler.command = None

    Handler.log_request(handler, "400")

    line = log_output.getvalue().strip()
    assert line.startswith("[webui] ")
    record = json.loads(line.removeprefix("[webui] "))
    assert record["method"] == "-"
    assert record["path"] == "-"
    assert record["status"] == 400
    assert record["remote"] == "-"


def test_log_request_includes_remote_address(log_output):
    handler = Handler.__new__(Handler)
    handler.command = "POST"
    handler.path = "/api/auth/login"
    handler.client_address = ("192.0.2.10", 54321)
    handler.headers = {}

    Handler.log_request(handler, "401")

    line = log_output.getvalue().strip()
    record = json.loads(line.removeprefix("[webui] "))
    assert record["remote"] == "192.0.2.10"
    assert "forwarded_for" not in record


def test_log_request_includes_resolved_client_ip_from_trusted_proxy(log_output, monkeypatch):
    """A trusted proxy's X-Forwarded-For resolves to a trustworthy client_ip.

    The socket peer here is loopback (a same-host reverse proxy), so the
    left-most-attacker-hop trap of the old first-hop read does not apply: the
    chain walk skips proxy hops and lands on the real client.
    """
    class Headers:
        def get(self, key, default=None):
            if key == "X-Forwarded-For":
                return "203.0.113.7"
            return default

        def get_all(self, key):
            if key == "X-Forwarded-For":
                return ["203.0.113.7"]
            return []

    handler = Handler.__new__(Handler)
    handler.command = "POST"
    handler.path = "/api/auth/login"
    handler.client_address = ("127.0.0.1", 54321)
    handler.headers = Headers()

    Handler.log_request(handler, "401")

    line = log_output.getvalue().strip()
    record = json.loads(line.removeprefix("[webui] "))
    assert record["remote"] == "127.0.0.1"
    assert record["client_ip"] == "203.0.113.7"


def test_log_request_ignores_forwarded_for_from_an_untrusted_peer(log_output):
    """A direct client's X-Forwarded-For must never become the logged client IP.

    Peer 192.0.2.10 is not a trusted proxy, so the header is attacker-chosen
    data: the trustworthy client IP falls back to the raw socket peer.
    """
    class Headers:
        def get(self, key, default=None):
            if key == "X-Forwarded-For":
                return "203.0.113.7, 198.51.100.9"
            return default

        def get_all(self, key):
            if key == "X-Forwarded-For":
                return ["203.0.113.7, 198.51.100.9"]
            return []

    handler = Handler.__new__(Handler)
    handler.command = "POST"
    handler.path = "/api/auth/login"
    handler.client_address = ("192.0.2.10", 54321)
    handler.headers = Headers()

    Handler.log_request(handler, "401")

    line = log_output.getvalue().strip()
    record = json.loads(line.removeprefix("[webui] "))
    assert record["remote"] == "192.0.2.10"
    assert record["client_ip"] == "192.0.2.10"
    # The raw chain is attacker-controlled and unbounded: dropped entirely for
    # an untrusted peer so a direct client cannot write arbitrary log volume
    # (the fail2ban-facing field stays the trustworthy raw peer).
    assert "forwarded_for_chain" not in record
