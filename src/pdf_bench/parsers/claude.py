"""Claude (Anthropic): each page is rendered to an image and read by the model."""

from __future__ import annotations

from pathlib import Path

from .. import config
from .base import entry
from .common import (
    DEFAULT_DPI,
    DEFAULT_MAX_PAGES,
    MAX_TOKENS,
    PAGE_PROMPT,
    Pages,
    b64,
    check_complete,
    int_option,
    render_pages,
    strip_fence,
)


def _anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=config.get("ANTHROPIC_API_KEY"))


def anthropic_pages(parser, path: Path) -> list[str]:
    client = _anthropic_client()
    model = parser.option("model")
    pages, used = Pages(), {"input_tokens": 0, "output_tokens": 0}
    for png in render_pages(path, dpi=int_option(parser, "dpi", DEFAULT_DPI),
                            max_pages=int_option(parser, "max_pages", DEFAULT_MAX_PAGES)):
        msg = client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(png)}},
                {"type": "text", "text": parser.option("prompt", PAGE_PROMPT)}]}])
        pages.append(strip_fence("".join(b.text for b in msg.content if getattr(b, "type", "") == "text")))
        check_complete(getattr(msg, "stop_reason", None), ("end_turn", "stop_sequence", None), "stop_reason", len(pages))
        u = getattr(msg, "usage", None)
        used["input_tokens"] += getattr(u, "input_tokens", 0) or 0
        used["output_tokens"] += getattr(u, "output_tokens", 0) or 0
    pages.raw = {"usage": used}
    return pages


entry("claude", "Claude (Anthropic)", "hosted_vlm", anthropic_pages, requires="anthropic",
       needs=("ANTHROPIC_API_KEY",), hosted=True, options={"model": "claude-sonnet-5-5", "dpi": 150},
       cannot={"link": "the model reads page images: a link target is not visible in the picture"},
       install="included", version="claude-sonnet-5-5", license="commercial API",
       source="https://docs.claude.com", reference="Claude Sonnet 4.6: in arXiv 2603.18652 (tables)",
       note="Page images sent to the Anthropic API. Other ids: claude-opus-5-5, claude-fable-5-1")
