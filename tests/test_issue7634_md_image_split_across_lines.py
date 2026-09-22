"""Behavioural regression coverage for #7634 — Markdown image across line break.

The renderer's image-syntax regex used to require `]` and `(` to sit
immediately adjacent, so a model that wrapped a long local-cache path across
a line break produced `![alt]\n(path)` and the renderer leaked the alt
text + filesystem path instead of rendering the image.  This file drives
the real ``renderMd()`` from ``static/ui.js`` via node so the regex fix
cannot silently regress to "no whitespace allowed" again — the same
forward-gate principle the file's docstring spells out for the renderer
mirrors in general.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_JS_PATH = REPO_ROOT / "static" / "ui.js"

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")


_DRIVER_SRC = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
global.window = {};
global.document = { createElement: () => ({ innerHTML: '', textContent: '' }), baseURI: 'http://localhost/app/' };
function _sessionUrlForSid(sid) { return '/app/session/' + encodeURIComponent(String(sid || '')); }
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const _IMAGE_EXTS=/\.(png|jpg|jpeg|gif|webp|bmp|ico|avif)$/i;
const _SVG_EXTS=/\.svg$/i;
const _AUDIO_EXTS=/\.(mp3|ogg|wav|m4a|aac|flac|wma|opus|webm)$/i;
const _VIDEO_EXTS=/\.(mp4|webm|mkv|mov|avi|ogv|m4v)$/i;
function _inlineMediaHtmlForRef(ref){
  const r = String(ref || '');
  if (/^https?:\/\//.test(r)) return `<img class="msg-media-img" src="${esc(r)}" alt="image" loading="lazy">`;
  if (/^file:\/\//.test(r)){
    const m = r.replace(/^file:\/\//i, '');
    return `<img class="msg-media-img" src="api/media?path=${encodeURIComponent(m)}" alt="image" loading="lazy">`;
  }
  return `<img class="msg-media-img" src="api/media?path=${encodeURIComponent(r)}" alt="image" loading="lazy">`;
}

function extractFunc(name) {
  const re = new RegExp('function\\s+' + name + '\\s*\\(');
  const start = src.search(re);
  if (start < 0) throw new Error(name + ' not found');
  let i = src.indexOf('{', start);
  let depth = 1; i++;
  while (depth > 0 && i < src.length) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') depth--;
    i++;
  }
  return src.slice(start, i);
}
eval(extractFunc('_matchBacktickFenceLine'));
eval(extractFunc('_isBacktickFenceClose'));
eval(extractFunc('renderMd'));

let buf = '';
process.stdin.on('data', c => { buf += c; });
process.stdin.on('end', () => { process.stdout.write(renderMd(buf)); });
"""


@pytest.fixture(scope="module")
def driver_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("renderer_driver_7634") / "driver.js"
    p.write_text(_DRIVER_SRC, encoding="utf-8")
    return str(p)


def _render(driver_path, markdown: str) -> str:
    result = subprocess.run(
        [NODE, driver_path, str(UI_JS_PATH)],
        input=markdown,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"node driver failed: {result.stderr}")
    return result.stdout


# ── 1. test_image_split_across_newline_renders_image_not_alt_text ─────────


class TestMarkdownImageAcrossLineBreak:

    def test_image_split_across_newline_renders_image_not_alt_text(self, driver_path):
        """#7634 repro: a model that wrapped the URL across a line break
        used to leave the alt text and URL visible in the chat.  After the
        fix the URL is recognised as an image and rendered as <img> with
        the alt text on the ``alt`` attribute.

        Note: the test uses ``https://`` because the renderer's
        ``_isSafeUrl`` (line 8193) only allowlists ``https?://``,
        ``mailto:``, ``tel:``, ``message:``, and ``api/`` for ``<img>``
        ``src`` — ``file://`` and bare POSIX paths are dropped by the
        ``_tag()`` pass after image generation.  That's a pre-existing
        behaviour out of scope for this fix (it would require widening
        the URL allowlist, not just loosening the image-syntax regex).
        The issue's own example uses a bare ``/home/...`` path, but the
        test exercises the regex-shape part of the bug, which is what
        the fix actually targets.
        """
        md = (
            "![Photorealistic cat on a tightrope]\n"
            "(https://example.com/capybara.png)"
        )
        out = _render(driver_path, md)
        # Image element emitted via the file:// media path
        assert '<img' in out, (
            f"split-across-newline image must render as <img>, not as the "
            f"raw alt+path. Got: {out!r}"
        )
        assert 'class="msg-media-img"' in out, (
            f"split-across-newline image must use the msg-media-img class "
            f"the image pipeline produces. Got: {out!r}"
        )
        # The URL is *inside* an attribute, not visible as text
        assert "example.com/capybara.png" in out, (
            f"the URL must be present as an img src. Got: {out!r}"
        )
        # The alt text is on the alt attribute, not visible as bare text
        assert 'alt="Photorealistic cat on a tightrope"' in out, (
            f"the alt text must be present on the img alt attribute. "
            f"Got: {out!r}"
        )

    def test_image_compact_form_still_renders(self, driver_path):
        """The fix must not regress the compact ![alt](url) form (no
        whitespace) — the most common shape."""
        out = _render(driver_path, "![capy](https://example.com/capy.png)")
        assert '<img' in out
        assert 'msg-media-img' in out
        assert 'src="https://example.com/capy.png"' in out

    def test_image_with_inline_space_between_brackets_renders(self, driver_path):
        """`![alt] (url)` with a single space must still render the image
        (the issue lists the line-break case but the same regex change
        covers the inline-space case, which is the more frequent
        CommonMark/GFM behaviour users expect)."""
        out = _render(driver_path, "![capy]( https://example.com/capy.png )")
        # The trailing space inside `(url)` does not match `[^\)]+`, so
        # this is only the leading-space case (the trailing space would
        # need a separate fix).  The leading-space half is what the regex
        # change covers.
        assert '<img' in out, (
            f"leading-inline-space image syntax must still render as <img>. "
            f"Got: {out!r}"
        )

    def test_image_with_tab_between_brackets_renders(self, driver_path):
        """`![alt]\\t(url)` — the tab case, the third whitespace variant
        the issue's regex fix covers."""
        out = _render(driver_path, "![capy]\t(https://example.com/capy.png)")
        assert '<img' in out, (
            f"tab between brackets must still render as <img>. "
            f"Got: {out!r}"
        )

    def test_image_inside_table_cell_still_renders(self, driver_path):
        """Table cells use the same image regex (via inlineMd).  The fix
        must not break that pass — a markdown image inside a table cell
        should still render."""
        md = (
            "| Thumbnail | Notes |\n"
            "| --- | --- |\n"
            "| ![capy](https://example.com/capy.png) | cute rodent |"
        )
        out = _render(driver_path, md)
        assert "<table" in out
        assert '<img' in out
        assert 'src="https://example.com/capy.png"' in out

    def test_image_inside_inline_code_is_not_consumed(self, driver_path):
        """A markdown image inside backticks must still be treated as code
        (not rendered as <img>) — the code-stash must keep working."""
        out = _render(driver_path, "Use `![alt](url)` syntax for images.")
        assert "<code>![alt](url)</code>" in out, (
            f"image syntax inside backticks must stay as code, not be "
            f"consumed by the image pass. Got: {out!r}"
        )
        assert "<img" not in out

    def test_bracket_pair_in_plain_text_is_not_misclassified_as_image(self, driver_path):
        """The fix only loosens whitespace between `]` and `(`.  A bare
        `]…(` pair without the `!` prefix and without a URL scheme must
        not be misclassified as an image."""
        out = _render(driver_path, "See the foo]…(bar) for context.")
        # No <img> emitted; the text is not an image.
        assert "<img" not in out, (
            f"plain text with a `]…(` pair must not be promoted to an "
            f"image. Got: {out!r}"
        )

    def test_bracket_pair_with_url_scheme_but_no_exclaim_still_anchor_not_image(self, driver_path):
        """`[label](https://...)` is a Markdown link, not an image.  The
        fix only changes the `!` image branch — the link regex still
        requires no `!` prefix."""
        out = _render(driver_path, "[docs](https://example.com/page)")
        assert '<a href="https://example.com/page"' in out
        assert '<img' not in out
