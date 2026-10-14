"""GPT (OpenAI, or any OpenAI-compatible server): each page is rendered to an image and read by the model."""

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


def _openai_client(base_url: str | None = None, api_key: str | None = None):
    import openai

    # the OpenAI key is only sent to OpenAI itself, never to another server (local servers, other gateways...)
    key = api_key or (config.get("OPENAI_API_KEY") if not base_url else "") or "EMPTY"
    return openai.OpenAI(base_url=base_url or None, api_key=key)


def openai_compat_pages(parser, path: Path, *, base_url: str | None = None, api_key: str | None = None) -> list[str]:
    """Chat-completions with an image per page: OpenAI itself, vLLM, Ollama, LM Studio..."""
    client = _openai_client(base_url or parser.option("base_url") or None, api_key)
    model = parser.option("model")
    limit = {"max_completion_tokens": MAX_TOKENS}
    pages, used = Pages(), {"input_tokens": 0, "output_tokens": 0}
    for png in render_pages(path, dpi=int_option(parser, "dpi", DEFAULT_DPI),
                            max_pages=int_option(parser, "max_pages", DEFAULT_MAX_PAGES)):
        resp = client.chat.completions.create(
            model=model, **limit,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": parser.option("prompt", PAGE_PROMPT)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(png)}"}}]}])
        pages.append(strip_fence(resp.choices[0].message.content or ""))
        check_complete(getattr(resp.choices[0], "finish_reason", None), ("stop", None), "finish_reason", len(pages))
        u = getattr(resp, "usage", None)
        used["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        used["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
    pages.raw = {"usage": used}
    return pages


entry("gpt", "GPT (OpenAI)", "hosted_vlm", openai_compat_pages, requires="openai",
       needs=("OPENAI_API_KEY",), hosted=True, options={"model": "gpt-6-astra", "dpi": 150},
       cannot={"link": "the model reads page images: a link target is not visible in the picture"},
       install="uv add openai", version="gpt-6-astra (id from OpenAI's model list: verify image input)",
       license="commercial API", source="https://developers.openai.com/api/docs/models/all",
       reference="OmniDocBench v1.6: 86.59 (GPT-5.2)",
       note="Same list: gpt-6.1-sol (cheaper), gpt-6-luna (cheapest). Image input was not confirmed on "
            "the page read")
