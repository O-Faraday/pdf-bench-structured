"""Gemini (Google): each page is rendered to an image and read by the model."""

from __future__ import annotations

from pathlib import Path

from .. import config
from .base import any_env, entry
from .common import (
    DEFAULT_DPI,
    DEFAULT_MAX_PAGES,
    PAGE_PROMPT,
    Pages,
    check_complete,
    int_option,
    render_pages,
    strip_fence,
)


def _gemini_client():
    from google import genai

    return genai.Client(api_key=config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY"))


def gemini_pages(parser, path: Path) -> list[str]:
    from google.genai import types

    client = _gemini_client()
    model = parser.option("model")
    pages, used = Pages(), {"input_tokens": 0, "output_tokens": 0}
    for png in render_pages(path, dpi=int_option(parser, "dpi", DEFAULT_DPI),
                            max_pages=int_option(parser, "max_pages", DEFAULT_MAX_PAGES)):
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=png, mime_type="image/png"), parser.option("prompt", PAGE_PROMPT)])
        pages.append(strip_fence(resp.text or ""))
        fin = getattr(resp.candidates[0], "finish_reason", None) if getattr(resp, "candidates", None) else None
        check_complete(getattr(fin, "name", fin), ("STOP", None), "finish_reason", len(pages))
        u = getattr(resp, "usage_metadata", None)
        used["input_tokens"] += getattr(u, "prompt_token_count", 0) or 0
        used["output_tokens"] += getattr(u, "candidates_token_count", 0) or 0
    pages.raw = {"usage": used}
    return pages


entry("gemini", "Gemini (Google)", "hosted_vlm", gemini_pages, requires="google.genai",
       hosted=True, options={"model": "gemini-3.8-flash", "dpi": 150},
       cannot={"link": "the model reads page images: a link target is not visible in the picture"},
       install="uv add google-genai", version="gemini-3.8-flash (id from a third-party list: verify)",
       license="commercial API", source="https://ai.google.dev/gemini-api/docs/models",
       reference="OmniDocBench v1.6: 92.91 (Gemini 3 Pro), 92.62 (Gemini 3 Flash)",
       check=any_env("GEMINI_API_KEY", "GOOGLE_API_KEY"),
       note="Also seen: gemini-3.1-pro-preview. Verify the id in Google's documentation")
