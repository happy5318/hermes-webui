"""Tests for issue #8017: a non-ASCII-only custom_providers name must not vanish.

``_custom_provider_slug_from_name()`` slugifies with an ASCII-only character
class, so a name made only of non-ASCII characters (e.g. a pure-CJK name)
slugifies to ``""``. The catalog builder treated that as "no provider" and
skipped the whole entry: its models never reached the WebUI model picker
while the CLI kept working — no error, no diagnostic.

The fix derives a deterministic ASCII identity for such records (explicit
``provider_key`` → endpoint-derived slug → short stable hash) so the entry
always has a stable identity that can round-trip as ``custom:<slug>``.
"""

from __future__ import annotations

import api.config as config


# ── the slugify primitive is unchanged ─────────────────────────────────────


def test_slug_from_name_still_empty_for_non_ascii_only_name():
    """The primitive keeps its documented behaviour; the fallback is separate."""
    assert config._custom_provider_slug_from_name("晨光鑫遇专用") == ""
    assert config._custom_provider_slug_from_name("") == ""


def test_slug_from_name_ascii_unchanged():
    assert config._custom_provider_slug_from_name("cgxy-cpa") == "custom:cgxy-cpa"
    assert config._custom_provider_slug_from_name("Local (127.0.0.1:15721)") == "custom:local-127.0.0.1-15721"


# ── the fallback ───────────────────────────────────────────────────────────


def test_fallback_prefers_explicit_provider_key():
    slug = config._named_custom_slug_fallback(
        {"provider_key": "cgxy-cpa", "base_url": "http://127.0.0.1:8317/v1"},
        "晨光鑫遇专用",
    )
    assert slug == "custom:cgxy-cpa"


def test_fallback_hashes_a_non_ascii_provider_key_deterministically():
    record = {"provider_key": "晨光鑫遇"}
    first = config._named_custom_slug_fallback(record, "晨光鑫遇专用")
    second = config._named_custom_slug_fallback(dict(record), "晨光鑫遇专用")
    assert first == second
    assert first.startswith("custom:")
    # Stable across processes: a content hash, not id()/repr()/random.
    assert first == config._named_custom_slug_fallback(record, "晨光鑫遇专用")


def test_fallback_uses_endpoint_slug_when_no_provider_key():
    slug = config._named_custom_slug_fallback(
        {"base_url": "http://127.0.0.1:8317/v1"},
        "晨光鑫遇专用",
    )
    # Prefer the hyphenated endpoint form: it stays a single safe identifier.
    assert slug == "custom:127.0.0.1-8317"


def test_fallback_hashes_the_name_when_nothing_else_is_available():
    slug = config._named_custom_slug_fallback({}, "晨光鑫遇专用")
    assert slug.startswith("custom:")
    assert len(slug) == len("custom:") + 8
    # Same name -> same slug; different name -> different slug.
    assert slug == config._named_custom_slug_fallback({}, "晨光鑫遇专用")
    assert slug != config._named_custom_slug_fallback({}, "别的名字")


def test_fallback_ignores_a_blank_provider_key():
    slug = config._named_custom_slug_fallback(
        {"provider_key": "   ", "base_url": "http://10.0.0.5:9000/v1"},
        "晨光鑫遇专用",
    )
    assert slug == "custom:10.0.0.5-9000"


# ── catalog builder wiring ─────────────────────────────────────────────────


def test_catalog_builder_falls_back_for_non_ascii_names():
    """The builder must call the fallback rather than dropping the entry."""
    src = open(config.__file__, encoding="utf-8").read()
    assert "_named_custom_slug_fallback(_cp, _cp_name)" in src
    # The fallback is only for entries whose name produced no slug at all.
    assert "if not _slug and _cp_name:" in src
