"""LlamaParse (LlamaCloud parse API v2)."""

from __future__ import annotations

from pathlib import Path

from .. import config
from .base import entry
from .common import Pages


def llamaparse_pages(parser, path: Path) -> list[str]:
    from llama_cloud import LlamaCloud

    tier = parser.option("tier", "agentic")
    if tier == "fast":
        raise ValueError("the 'fast' tier returns no Markdown: use cost_effective, agentic or agentic_plus")
    client = LlamaCloud(api_key=config.get("LLAMA_CLOUD_API_KEY"))
    file = client.files.create(file=Path(path), purpose="parse")
    result = client.parsing.parse(file_id=file.id, tier=tier, version=parser.option("version", "latest"),
                                  expand=["markdown"])
    pages = Pages([p.markdown for p in result.markdown.pages])
    pages.raw = result.model_dump() if hasattr(result, "model_dump") else None
    return pages


entry("llamaparse", "LlamaParse (LlamaCloud)", "cloud_service", llamaparse_pages, requires="llama_cloud",
       hosted=True, needs=("LLAMA_CLOUD_API_KEY",), install='uv add "llama-cloud>=2.8"',
       options={"tier": "agentic", "version": "latest"}, version="llama-cloud >= 2.8, parse API v2",
       license="commercial (LlamaIndex)", source="https://developers.llamaindex.ai/llamaparse/parse/getting_started/",
       note="Tiers: cost_effective, agentic (default), agentic_plus; 'fast' returns no Markdown. 'version' latest "
            "changes over time: pin a version for reproducible runs. Billed per page, more for higher tiers")
