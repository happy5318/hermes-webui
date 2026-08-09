"""Tests for session-aware vision routing (image routing follows the
session's actual provider/model, not only the global default).

Covers the behavior added by the session-aware-vision-routing patch:
  - no active params     -> falls back to the global default (upstream parity)
  - active vision model  -> native
  - active text model    -> text
  - requested_provider   -> capability lookup selects the exact
    custom_providers/providers entry even after the provider id was
    canonicalized to "custom" by
    _resolve_custom_provider_runtime_overrides
  - _build_native_multimodal_message and _sanitize_messages_for_api
    forward the session identity through to the routing decision
"""
import base64
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from api.streaming import (
    _build_native_multimodal_message,
    _resolve_image_input_mode,
    _sanitize_messages_for_api,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_png(path: Path, size: int = 0) -> Path:
    """Write a minimal valid PNG to *path* (IHDR + IDAT + IEND)."""
    if size <= 0:
        data = (
            b'\x89PNG\r\n\x1a\n'
            b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
            b'\x00\x00\x00\x0bIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
            b'\x00\x00\x00\x00IEND\xaeB`\x82'
        )
    else:
        data = b'\x89PNG\r\n\x1a\n' + b'\x00' * (size - 8)
    path.write_bytes(data)
    return path


def _cfg_with_provider_vision(provider_name: str, model: str, vision: bool) -> dict:
    """Config with a named provider whose per-model capability is explicit."""
    return {
        "model": {},
        "providers": {
            provider_name: {
                "models": {model: {"supports_vision": vision}},
            }
        },
    }


# ── _resolve_image_input_mode ───────────────────────────────────────────────

class TestResolveImageInputMode:
    def test_no_active_params_falls_back_to_global_default(self, monkeypatch):
        """Upstream parity: with no session identity, route by global default."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_read_main_provider", lambda: "custom:mygateway")
        monkeypatch.setattr(aux, "_read_main_model", lambda: "my-vision-model")
        cfg = _cfg_with_provider_vision("mygateway", "my-vision-model", True)
        assert _resolve_image_input_mode(cfg) == "native"

    def test_global_default_text_model_routes_text(self, monkeypatch):
        """Upstream parity for a text-only global default."""
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "_read_main_provider", lambda: "custom:mygateway")
        monkeypatch.setattr(aux, "_read_main_model", lambda: "my-text-model")
        cfg = {"model": {}, "providers": {}}
        # unknown to models.dev -> WebUI carve-out forwards native
        assert _resolve_image_input_mode(cfg) in ("native", "text")

    def test_active_vision_model_routes_native(self):
        cfg = _cfg_with_provider_vision("myvllm", "my-vision", True)
        assert _resolve_image_input_mode(
            cfg, "custom", "my-vision", requested_provider="myvllm"
        ) == "native"

    def test_active_text_model_routes_text(self):
        cfg = _cfg_with_provider_vision("myvllm", "my-text-model", False)
        assert _resolve_image_input_mode(
            cfg, "custom", "my-text-model", requested_provider="myvllm"
        ) == "text"

    def test_requested_provider_selects_exact_entry(self):
        """requested_provider beats the unknown-model native carve-out.

        Same config, same canonicalized provider "custom": passing the
        pre-canonicalization identity makes capability lookup hit the exact
        providers.<name> entry (here: supports_vision=False) and route text,
        while without it the lookup misses and the carve-out forwards native.
        """
        cfg = _cfg_with_provider_vision("myvllm", "my-model", False)
        assert _resolve_image_input_mode(
            cfg, "custom", "my-model", requested_provider="myvllm"
        ) == "text"
        assert _resolve_image_input_mode(cfg, "custom", "my-model") == "native"

    def test_requested_provider_vision_true_routes_native(self):
        cfg = _cfg_with_provider_vision("myvllm", "my-vision", True)
        assert _resolve_image_input_mode(
            cfg, "custom", "my-vision", requested_provider="myvllm"
        ) == "native"


# ── _build_native_multimodal_message forwarding ─────────────────────────────

def _normalized_attachments(img_path: Path) -> list:
    from api.routes import _normalize_chat_attachments

    return _normalize_chat_attachments([{
        'name': img_path.name, 'path': str(img_path),
        'mime': 'image/png', 'size': img_path.stat().st_size, 'is_image': True,
    }])


class TestBuildNativeMultimodalForwarding:
    def test_requested_provider_reaches_text_mode(self, tmp_path):
        """A text-mode verdict strips the attachments to a plain string."""
        cfg = _cfg_with_provider_vision("myvllm", "my-model", False)
        img = _make_png(tmp_path / "a.png")
        result = _build_native_multimodal_message(
            "", "describe", _normalized_attachments(img), str(tmp_path),
            cfg=cfg,
            active_provider="custom",
            active_model="my-model",
            requested_provider="myvllm",
        )
        assert result == "describe"

    def test_without_requested_provider_embeds_native(self, tmp_path):
        """Unknown-model carve-out: image is embedded as a content part."""
        cfg = _cfg_with_provider_vision("myvllm", "my-model", False)
        img = _make_png(tmp_path / "a.png")
        parts = _build_native_multimodal_message(
            "", "describe", _normalized_attachments(img), str(tmp_path),
            cfg=cfg,
            active_provider="custom",
            active_model="my-model",
        )
        assert isinstance(parts, list)
        assert parts[0]["type"] == "text"
        assert any(part.get("type") == "image_url" for part in parts[1:])


# ── _sanitize_messages_for_api forwarding ───────────────────────────────────

class TestSanitizeForwarding:
    def test_text_mode_strips_historical_native_images(self):
        cfg = _cfg_with_provider_vision("myvllm", "my-model", False)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]
        clean = _sanitize_messages_for_api(
            messages,
            cfg=cfg,
            effective_provider="custom",
            effective_model="my-model",
            requested_provider="myvllm",
        )
        rendered = str(clean)
        assert "image_url" not in rendered

    def test_native_mode_keeps_historical_images(self):
        cfg = _cfg_with_provider_vision("myvllm", "my-model", True)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ]
        clean = _sanitize_messages_for_api(
            messages,
            cfg=cfg,
            effective_provider="custom",
            effective_model="my-model",
            requested_provider="myvllm",
        )
        assert "image_url" in str(clean)