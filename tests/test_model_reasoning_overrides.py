"""Per-model reasoning-effort override resolution (config → coercion).

Covers the shared resolver ``configured_reasoning_effort_for_model()`` plus the
Gateway request path, which must never hand the Hermes Gateway transport
address to a provider capability probe.
"""

import pytest

from api import config as cfg
from api import gateway_chat


def test_configured_reasoning_effort_prefers_model_override():
    config_data = {
        "agent": {
            "reasoning_effort": "max",
            "reasoning_overrides": {
                "gemini-3.6-flash-tiered": "low",
            },
        },
    }

    assert cfg.configured_reasoning_effort_for_model(
        config_data,
        model_id="gemini-3.6-flash-tiered",
        provider_id="custom:example-gateway",
    ) == "low"


def test_configured_reasoning_effort_keeps_global_for_unmatched_model():
    config_data = {
        "agent": {
            "reasoning_effort": "high",
            "reasoning_overrides": {
                "gemini-3.6-flash-tiered": "low",
            },
        },
    }

    assert cfg.configured_reasoning_effort_for_model(
        config_data,
        model_id="claude-sonnet-4-6",
        provider_id="custom:example-gateway",
    ) == "high"


@pytest.mark.parametrize(
    "model_id",
    [
        "@openrouter:gemini-3.6-flash-tiered",  # qualified companion form
        "GEMINI-3.6-FLASH-TIERED",              # case-insensitive
    ],
)
def test_configured_reasoning_effort_matches_normalized_model_ids(model_id):
    """Override keys must survive qualified/cased model identifiers.

    The WebUI receives model ids in several shapes (``@provider:model`` from the
    picker, raw upstream casing from a custom gateway). A stored override keyed
    on the plain lowercase id must still win over the global value.
    """
    config_data = {
        "agent": {
            "reasoning_effort": "max",
            "reasoning_overrides": {
                "gemini-3.6-flash-tiered": "low",
            },
        },
    }

    assert cfg.configured_reasoning_effort_for_model(
        config_data,
        model_id=model_id,
        provider_id="custom:example-gateway",
    ) == "low"


def _install_lmstudio_probe_recorder(monkeypatch, *, options):
    """Record every base_url handed to the LM Studio capability probe."""
    seen: list[str | None] = []

    def _fake_probe(model, base_url, *, api_key=None, timeout=5.0):
        seen.append(base_url)
        return list(options)

    monkeypatch.setattr(cfg, "_lmstudio_model_reasoning_options", _fake_probe)
    return seen


def test_gateway_reasoning_effort_probes_provider_endpoint_not_gateway(monkeypatch):
    """Gateway requests must probe the LM Studio endpoint, never the Gateway.

    ``_gateway_base_url()`` is the Hermes Gateway transport address (where
    WebUI POSTs ``/v1/chat/completions``). Forwarding it as the provider
    capability endpoint made ``resolve_model_reasoning_efforts()`` probe the
    Gateway as though it were LM Studio, coercing the configured override
    against the wrong capability set.
    """
    gateway_url = "http://127.0.0.1:8642"
    lmstudio_url = "http://192.168.1.50:1234/v1"

    # Configured LM Studio endpoint, distinct from the Gateway transport.
    monkeypatch.setitem(cfg.cfg, "providers", {"lmstudio": {"base_url": lmstudio_url}})
    # LM Studio advertises a ladder that tops out below the configured "max".
    seen = _install_lmstudio_probe_recorder(
        monkeypatch, options=["low", "medium", "high"]
    )

    config_data = {
        "webui_gateway_base_url": gateway_url,
        "agent": {
            "reasoning_effort": "low",
            "reasoning_overrides": {"local-thinker": "max"},
        },
    }

    effort = gateway_chat._gateway_reasoning_effort_for_request(
        config_data,
        model="local-thinker",
        model_provider="lmstudio",
    )

    assert seen, "LM Studio capability probe was never invoked"
    normalized = [cfg._normalize_base_url_for_match(url) for url in seen]
    assert cfg._normalize_base_url_for_match(gateway_url) not in normalized, (
        f"Gateway transport address leaked into the provider probe: {seen}"
    )
    assert normalized == [cfg._normalize_base_url_for_match(lmstudio_url)]
    # The per-model "max" override must clamp down to the probed ceiling.
    assert effort == "high"


def test_gateway_reasoning_effort_ignores_gateway_url_for_non_probed_provider(
    monkeypatch,
):
    """Non-probed providers keep the resolved override untouched."""
    config_data = {
        "webui_gateway_base_url": "http://127.0.0.1:8642",
        "agent": {
            "reasoning_effort": "max",
            "reasoning_overrides": {"gemini-3.6-flash-tiered": "low"},
        },
    }

    assert gateway_chat._gateway_reasoning_effort_for_request(
        config_data,
        model="gemini-3.6-flash-tiered",
        model_provider="custom:example-gateway",
    ) == "low"


def test_configured_reasoning_effort_qualified_custom_provider_models_resolve_consistently():
    """Qualified custom-provider models must resolve overrides consistently across paths."""
    config_data = {
        "agent": {
            "reasoning_effort": "high",
            "reasoning_overrides": {
                "claude-opus-4.5": "low",
            },
        },
    }

    # Native / shared config resolver
    resolved_native = cfg.configured_reasoning_effort_for_model(
        config_data,
        model_id="@custom:example-gateway:claude-opus-4-5",
    )
    assert resolved_native == "low"

    # Gateway path
    resolved_gateway = gateway_chat._gateway_reasoning_effort_for_request(
        config_data,
        model="@custom:example-gateway:claude-opus-4-5",
        model_provider="custom:example-gateway",
    )
    assert resolved_gateway == "low"


def test_configured_reasoning_effort_unrepresentable_override_retains_global():
    """An unrepresentable per-model override must retain the working global effort."""
    config_data = {
        "agent": {
            "reasoning_effort": "high",
            "reasoning_overrides": {
                "some-model": "ultra",
            },
        },
    }

    resolved = cfg.configured_reasoning_effort_for_model(
        config_data,
        model_id="some-model",
    )
    assert resolved == "high"

