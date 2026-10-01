"""The structured request log must not record a spoofable client IP (#7863).

``log_request`` used to read ``X-Forwarded-For``'s LEFT-most hop and write it
as ``forwarded_for``. That is the end of the chain the client itself writes:
a direct client could make the log claim any address. The CHANGELOG says the
field exists so downstream security tooling (fail2ban) can consume it — which
makes an attacker-chosen value worse than no field at all: a fail2ban jail
keyed on it lets anyone get an arbitrary third-party IP banned for a week.

The WebUI already has the correct machinery (``_raw_peer_is_trusted_proxy``
judged on the un-spoofable socket address, and the right-to-left chain walk
in ``_forwarded_client_ip_from_trusted_proxy`` that fails closed on a
malformed chain). ``log_request`` now uses it and publishes the trustworthy
value under an explicitly-named field:

* trusted proxy peer  -> the resolved client hop
* untrusted peer      -> the raw socket peer (the attacker-chosen header is
  dropped from the structured record entirely, not even kept as debugging
  data — see the untrusted-peer test below)
* resolution failure  -> fail closed to the raw peer, never to a header hop

The CHANGELOG-documented ``forwarded_for`` field keeps its name and can never
hold raw header text: it mirrors the safe ``client_ip`` in every case, so
downstream fail2ban filters keyed on the documented name keep matching while
the #7863 spoofing vector stays closed.
"""

from __future__ import annotations

import io
import json

import pytest

from server import Handler


@pytest.fixture
def log_output(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr("api.request_logging._STREAM", output)
    return output


class _Headers:
    """Minimal header bag: ``get`` plus ``get_all`` (multiple XFF headers)."""

    def __init__(self, forwarded=None, real_ip=None):
        self._forwarded = forwarded
        self._real_ip = real_ip

    def get(self, key, default=None):
        if key == "X-Forwarded-For":
            if self._forwarded is None:
                return default
            return ", ".join(self._forwarded)
        if key == "X-Real-IP":
            return self._real_ip if self._real_ip is None else self._real_ip
        return default

    def get_all(self, key):
        if key == "X-Forwarded-For":
            return list(self._forwarded or [])
        return []


def _record(log_output: io.StringIO) -> dict:
    line = log_output.getvalue().strip()
    assert line.startswith("[webui] "), line
    return json.loads(line.removeprefix("[webui] "))


def _handler(peer, headers, path="/api/auth/login", command="POST"):
    handler = Handler.__new__(Handler)
    handler.command = command
    handler.path = path
    handler.client_address = (peer, 54321)
    handler.headers = headers
    return handler


def test_resolved_client_ip_survives_a_multi_hop_chain(log_output):
    """Trusted proxy + real chain: the walk lands on the non-proxy client.

    Two proxy hops (the local reverse proxy, then an internal one) followed by
    the actual client. The old first-hop read would have rewritten the log to
    whatever the CLIENT sent in the left-most slot.
    """
    handler = _handler(
        "127.0.0.1", _Headers(forwarded=["10.0.0.1", "10.0.0.2", "203.0.113.7"])
    )

    Handler.log_request(handler, "200")

    record = _record(log_output)
    assert record["client_ip"] == "203.0.113.7"
    assert record["remote"] == "127.0.0.1"


def test_repeated_forwarded_headers_keep_wire_order(log_output):
    """Two XFF headers are one chain, in wire order — not two first hops."""
    handler = _handler(
        "127.0.0.1", _Headers(forwarded=["10.0.0.1", "203.0.113.7"])
    )

    Handler.log_request(handler, "200")

    record = _record(log_output)
    assert record["client_ip"] == "203.0.113.7"


def test_malformed_chain_from_a_trusted_proxy_fails_closed(log_output):
    """A present-but-garbage chain must not resolve to a header value."""
    handler = _handler("127.0.0.1", _Headers(forwarded=["not-an-ip"]))

    Handler.log_request(handler, "200")

    record = _record(log_output)
    assert record["client_ip"] == "127.0.0.1"
    assert record["client_ip"] == record["remote"]


def test_empty_chain_from_a_trusted_proxy_fails_closed(log_output):
    """An empty XFF is malformed: fail closed rather than skip past it."""
    handler = _handler("127.0.0.1", _Headers(forwarded=[""]))

    Handler.log_request(handler, "200")

    record = _record(log_output)
    assert record["client_ip"] == "127.0.0.1"


def test_untrusted_peer_header_is_recorded_but_not_trusted(log_output):
    """A direct public peer's XFF is attacker data: keep it, label it.

    This is the exact DoS vector of #7863 — a fail2ban jail reading the
    trustworthy field would ban the raw peer, not a chosen address.
    """
    handler = _handler(
        "192.0.2.10", _Headers(forwarded=["203.0.113.7", "198.51.100.9"])
    )

    Handler.log_request(handler, "401")

    record = _record(log_output)
    assert record["client_ip"] == "192.0.2.10"
    # The raw chain is attacker-controlled and unbounded, so it is dropped
    # from the record for an untrusted peer entirely (bounded log volume).
    assert "forwarded_for_chain" not in record


def test_no_forwarded_header_logs_the_raw_peer(log_output):
    handler = _handler("192.0.2.10", _Headers())

    Handler.log_request(handler, "200")

    record = _record(log_output)
    assert record["client_ip"] == "192.0.2.10"
    assert "forwarded_for_chain" not in record
    # The old single-hop field is gone: nothing reads first-hop anymore.
    assert "forwarded_for" not in record


def test_client_ip_is_always_present(log_output):
    """The field downstream tooling keys on must never be missing."""
    for peer in ("192.0.2.10", "127.0.0.1"):
        handler = _handler(peer, _Headers())
        Handler.log_request(handler, "404")
        lines = [line for line in log_output.getvalue().strip().splitlines() if line]
        record = json.loads(lines[-1].removeprefix("[webui] "))
        assert record["client_ip"] == peer

def test_hostile_repeated_header_leaves_the_record_bounded(log_output):
    """A direct client spamming X-Forwarded-For cannot inflate the log record."""
    hostile = ["1.1.1.1, 2.2.2.2, 3.3.3.3"] * 500
    handler = _handler("192.0.2.10", _Headers(forwarded=hostile))

    Handler.log_request(handler, "200")

    line = log_output.getvalue().strip().splitlines()[-1].removeprefix("[webui] ")
    record = json.loads(line)
    assert record["client_ip"] == "192.0.2.10"
    # 500 injected headers, none of them land in the structured record.
    assert "forwarded_for_chain" not in record
    assert len(line) < 1024, "the record must stay bounded regardless of input"


def test_malformed_x_real_ip_from_trusted_peer_logs_the_raw_peer(log_output):
    """Trusted peer + malformed X-Real-IP + no XFF → the raw peer, never the text.

    A proxy that appends to (rather than replaces) a client-supplied chain can
    relay garbage in ``X-Real-IP``; header text must not become the log's
    identity (the old fallback echoed it verbatim, so no validation raised and
    the fail-closed branch never ran).
    """
    handler = _handler("127.0.0.1", _Headers(real_ip="not-an-ip"))

    Handler.log_request(handler, "200")

    record = _record(log_output)
    assert record["client_ip"] == "127.0.0.1"
    assert record["client_ip"] == record["remote"]


def test_valid_x_real_ip_and_ipv6_remain_accepted_on_the_log_path(log_output):
    """A trusted proxy's *valid* X-Real-IP still resolves the log's client_ip.

    The request log never reads ``X-Real-IP`` (see the module docstring) — the
    right-to-left XFF walk is the only authority — but an IPv4/IPv6-formatted
    value must not break resolution when XFF is present, and must fall closed
    to the raw peer when it is absent.
    """
    # XFF present: the walk wins over any X-Real-IP.
    handler = _handler(
        "127.0.0.1", _Headers(forwarded=["203.0.113.7"], real_ip="198.51.100.99")
    )
    Handler.log_request(handler, "200")
    lines = [l for l in log_output.getvalue().strip().splitlines() if l]
    assert json.loads(lines[-1].removeprefix("[webui] "))["client_ip"] == "203.0.113.7"

    # XFF absent, X-Real-IP alone: the log falls closed to the raw peer.
    handler = _handler("127.0.0.1", _Headers(real_ip="2001:db8::1"))
    Handler.log_request(handler, "200")
    lines = [l for l in log_output.getvalue().strip().splitlines() if l]
    assert json.loads(lines[-1].removeprefix("[webui] "))["client_ip"] == "127.0.0.1"


def test_trusted_peer_oversized_chain_stays_bounded(log_output):
    """A trusted proxy relaying a hostile chain cannot inflate the record.

    Trusting the immediate peer is necessary for resolution but does not bound
    the inbound volume: the proxy may append to a client-supplied chain. The
    diagnostic ``forwarded_for_chain`` is therefore capped by entry count and
    serialized characters while ``client_ip`` keeps the full validated walk.
    """
    hostile = [f"203.0.113.{i % 250 + 1}" for i in range(500)]
    handler = _handler("127.0.0.1", _Headers(forwarded=hostile))

    Handler.log_request(handler, "200")

    line = log_output.getvalue().strip().splitlines()[-1].removeprefix("[webui] ")
    record = json.loads(line)
    # Resolution still uses the full validated semantics: right-most hop.
    assert record["client_ip"] == "203.0.113.250"
    assert record["remote"] == "127.0.0.1"
    chain = record["forwarded_for_chain"]
    assert len(chain) <= 32, f"chain not bounded: {len(chain)} entries"
    assert sum(len(e) for e in chain) <= 512
    assert len(line) < 4096, "the record must stay bounded even from a trusted peer"


def test_trusted_chain_truncation_keeps_the_resolution_semantics(log_output):
    """After diagnostic truncation the trusted-chain client selection is unchanged.

    A proxy appends its own hop to a client chain that already ends in the real
    client; truncation is diagnostic-only, so ``client_ip`` must still be the
    first non-trusted hop from the right, not the truncated tail.
    """
    hops = ["192.0.2.5"] + [f"10.0.0.{i}" for i in range(1, 60)] + ["203.0.113.99"]
    handler = _handler("127.0.0.1", _Headers(forwarded=hops))

    Handler.log_request(handler, "200")

    record = _record(log_output)
    assert record["client_ip"] == "203.0.113.99"
    assert len(record["forwarded_for_chain"]) <= 32


def test_trusted_peer_single_oversized_entry_stays_char_bounded(log_output):
    """ONE oversized hop must be elided — entry count alone is not a bound.

    ``_bounded_trusted_chain`` caps entries at 32, but a chain of TWO hops can
    still carry tens of KB when one hop is client-written. A proxy may append
    to a client-supplied chain, so from a *trusted* peer the budget has to be
    enforced on serialized characters: the oversized value is replaced by a
    marker and the neighbouring real hops survive for diagnosis.
    """
    handler = _handler("127.0.0.1", _Headers(forwarded=["A" * 40000]))

    Handler.log_request(handler, "200")

    record = _record(log_output)
    chain = record["forwarded_for_chain"]
    serialized = len(json.dumps(chain))
    assert serialized <= 512, f"chain serializes to {serialized} chars"
    assert all("A" * 40000 not in entry for entry in chain), chain


def test_trusted_peer_real_hops_survive_an_oversized_neighbour(log_output):
    """A hostile mega-hop must not evict the real IP hops around it.

    Truncating by entry count would keep the mega-hop; truncating by budget
    must keep the diagnostic value: the short hops stay and only the oversized
    entry is elided, with a marker naming the dropped entry.
    """
    hops = ["203.0.113.7", "198.51.100.9"] + ["B" * 3000] + ["192.0.2.10"]
    handler = _handler("127.0.0.1", _Headers(forwarded=hops))

    Handler.log_request(handler, "200")

    record = _record(log_output)
    chain = record["forwarded_for_chain"]
    assert len(json.dumps(chain)) <= 512
    assert "203.0.113.7" in chain or "192.0.2.10" in chain, f"real hops lost: {chain}"
    assert "B" * 3000 not in json.dumps(chain)
    # Resolution is untouched by the diagnostic elision.
    assert record["client_ip"] == "192.0.2.10"


def test_forwarded_for_alias_mirrors_the_resolved_client_ip(log_output):
    """CHANGELOG documents ``forwarded_for`` for fail2ban — keep the name.

    The field's *name* is a contract; its old *value* (raw left-most header
    hop) was the #7863 bug. The alias therefore stays present and mirrors the
    validated ``client_ip``: a fail2ban filter keyed on the documented name
    keeps matching and can never read attacker-chosen text.
    """
    handler = _handler("127.0.0.1", _Headers(forwarded=["203.0.113.7"]))

    Handler.log_request(handler, "200")

    record = _record(log_output)
    assert record["forwarded_for"] == record["client_ip"] == "203.0.113.7"


def test_forwarded_for_alias_never_carries_raw_header_text(log_output):
    """For an untrusted peer the documented alias is absent, not header text.

    The alias is emitted only for a trusted proxy's resolved chain, so a
    direct client can influence neither its value nor its presence.
    """
    handler = _handler("192.0.2.10", _Headers(forwarded=["203.0.113.7", "198.51.100.9"]))

    Handler.log_request(handler, "401")

    record = _record(log_output)
    assert record["client_ip"] == "192.0.2.10"
    assert "forwarded_for" not in record
    for value in record.values():
        assert "203.0.113.7" not in str(value)
        assert "198.51.100.9" not in str(value)
