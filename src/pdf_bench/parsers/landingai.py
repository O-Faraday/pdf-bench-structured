"""LandingAI Agentic Document Extraction (ADE): typed chunks written out as Markdown with HTML tables."""

from __future__ import annotations

import re
from pathlib import Path

from .base import entry
from .common import ANCHOR, Pages

_ADE_WRAP = re.compile(r"<::(.*?)::>", re.S)
_ADE_KIND = re.compile(r":\s*(figure|logo|card|attestation|scan_code)\s*$", re.I)
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(/|of)\s*\d+)?$", re.I)


def _ade_unwrap(markdown: str) -> str:
    """The text inside an image-based chunk's ``<::description: figure::>`` wrapper (the wrapper is not text)."""
    inner = [_ADE_KIND.sub("", m.strip()).strip() for m in _ADE_WRAP.findall(markdown)]
    outside = _ADE_WRAP.sub(" ", markdown).strip()
    return "\n\n".join(x for x in (outside, *inner) if x)


def _ade_running(chunks) -> set[str]:
    """Margin texts that repeat on several pages once the digits are ignored: the running headers and footers."""
    pages: dict[str, set[int]] = {}
    for ch in chunks:
        if getattr(ch, "type", "") == "marginalia":
            key = re.sub(r"\d+", "#", ANCHOR.sub("", ch.markdown or "")).strip().lower()
            page = int(getattr(getattr(ch, "grounding", None), "page", 0) or 0)
            pages.setdefault(key, set()).add(page)
    return {k for k, v in pages.items() if len(v) >= 2}


def landingai_chunks_to_pages(chunks) -> list[str]:
    """Markdown per page from LandingAI ADE chunks (``id``, ``type``, ``markdown``, ``grounding.page``).

    A ``marginalia`` chunk (the margins of the page) is left out only when it is a page number or repeats on
    several pages (running header or footer); any other one is kept, because a footnote at the foot of the
    page is margin text too. Image-based chunks arrive as ``<::description: figure::>``: a figure keeps its
    description (the caption is part of the chunk, see the ADE markdown response page) after an ``[image]``
    marker, a signature block (``attestation``) gives its readable text and a ``[signature]`` marker.
    """
    by_page: dict[int, list[str]] = {}
    running = _ade_running(chunks)
    for ch in chunks:
        page = int(getattr(getattr(ch, "grounding", None), "page", 0) or 0)
        by_page.setdefault(page, [])
        kind = getattr(ch, "type", "")
        md = ANCHOR.sub("", ch.markdown or "").strip()
        if kind == "marginalia":
            key = re.sub(r"\d+", "#", md).lower()
            if _PAGE_NUMBER.match(md.strip()) or key in running:
                continue
            text = md
        elif kind == "attestation":
            m = re.search(r"Readable Text:\s*(.+)", md)
            text = (m.group(1).strip() + "\n\n" if m else "") + "[signature]"
        elif kind in ("figure", "logo", "card", "scan_code"):
            text = "[image]" + (("\n\n" + d) if (d := _ade_unwrap(md)) else "")
        else:
            text = md
        if text:
            by_page[page].append(text)
    if not by_page:
        return []
    return ["\n\n".join(by_page.get(i, [])) for i in range(max(by_page) + 1)]


def landingai_pages(parser, path: Path) -> list[str]:
    from landingai_ade import LandingAIADE

    client = LandingAIADE(environment="eu") if parser.option("region", "us") == "eu" else LandingAIADE()
    resp = client.parse(document=path, model=parser.option("model", "dpt-2-latest"))
    pages = Pages(landingai_chunks_to_pages(resp.chunks) or [resp.markdown])
    pages.raw = resp.model_dump() if hasattr(resp, "model_dump") else None
    return pages


entry("landingai_ade", "LandingAI Agentic Document Extraction", "cloud_service", landingai_pages,
       requires="landingai_ade", hosted=True, needs=("VISION_AGENT_API_KEY",), install="uv add landingai-ade",
       options={"model": "dpt-2-latest", "region": "us"}, version="landingai-ade 1.18.1, model dpt-2-latest, adapter 2",
       license="commercial (LandingAI)", source="https://docs.landing.ai/ade/ade-python",
       reference="none found from a neutral source",
       note="Pinned snapshots exist (dpt-2-20260410, dpt-2-20260903...): pin one for reproducible runs. "
            "PDF_BENCH_LANDINGAI_ADE_REGION=eu for an EU key. Marginalia chunks (the margins of the page) "
            "are kept unless they are page numbers or repeat on several pages (a footnote is margin text too). A figure is "
            "written as [image] followed by its description/caption; a signature block gives its readable text and [signature]. "
            "Heading levels are not provided (all headings come as '#')")
