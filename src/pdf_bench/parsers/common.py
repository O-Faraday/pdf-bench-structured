"""Shared by the parsers that send page images to a model, and by the Markdown writers of the services."""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

PAGE_PROMPT = (
    "Convert this document page to Markdown.\n"
    "- Keep the natural reading order, including across columns.\n"
    "- Render every table as an HTML <table> with <th> header cells and rowspan/colspan where needed.\n"
    "- Write formulas as LaTeX.\n"
    "- Leave out running page headers, footers and page numbers.\n"
    "- Do not describe images; write [image] where a figure is, then its caption as plain text.\n"
    "- Keep headings as Markdown headings with the right level, lists as lists (with nesting), "
    "links as [text](url) and footnotes as [^n]: text.\n"
    "- Write a form field as 'Label: value' and a checkbox as ☒ label (ticked) or ☐ label (empty).\n"
    "Return only the converted content, with no comment."
)

class Pages(list):
    """Pages of Markdown that remember the service's raw response (shown next to the scores)."""

    raw: object = None

DEFAULT_DPI = 150

DEFAULT_MAX_PAGES = 30

MAX_TOKENS = 8192

def render_pages(path: Path, *, dpi: int = DEFAULT_DPI, max_pages: int = DEFAULT_MAX_PAGES) -> list[bytes]:
    """PNG bytes of each page (at most ``max_pages``)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        out = []
        for i in range(min(len(pdf), max_pages)):
            image = pdf[i].render(scale=dpi / 72).to_pil()
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
    finally:
        pdf.close()

def b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")

def int_option(parser, key: str, default: int) -> int:
    try:
        return int(parser.option(key, str(default)))
    except ValueError:
        return default

def strip_fence(text: str) -> str:
    """Models often wrap the whole answer in a ```markdown fence: remove it."""
    t = text.strip()
    m = re.fullmatch(r"```(?:markdown|md|html)?\s*\n(.*?)\n```", t, flags=re.S)
    return (m.group(1) if m else t).strip()

def check_complete(reason, ok: tuple, name: str, page: int) -> None:
    """A page cut off by the token limit (or refused) would be scored as if the model had missed what follows."""
    if reason not in ok:
        raise RuntimeError(f"page {page} of the answer is incomplete ({name} = {reason}): not scored")

ANCHOR = re.compile(r"<a id=['\"][^'\"]*['\"]></a>\s*")
