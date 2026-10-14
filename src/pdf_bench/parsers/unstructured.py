"""Unstructured ``partition_pdf``: typed elements written out as Markdown."""

from __future__ import annotations

from pathlib import Path

from .base import entry
from .common import Pages


def _field(el, name, default=None):
    """Attribute or key of an element: the library gives objects, a saved response gives dicts."""
    return el.get(name, default) if isinstance(el, dict) else getattr(el, name, default)


def unstructured_elements_to_pages(elements) -> list[str]:
    """Markdown/HTML per page from ``unstructured`` elements.

    ``Title`` becomes a heading (level = ``category_depth`` + 1), consecutive ``ListItem`` a bullet list,
    ``Table`` its ``text_as_html``, ``Image`` ``[image]``. Running ``Header``, ``Footer`` and ``PageNumber`` are
    left out. The library has no footnote element, so footnotes arrive as ordinary text.
    """
    by_page: dict[int, list[str]] = {}
    items: list[str] = []  # a run of list items, flushed into the current page
    items_page = 1

    def flush():
        if items:
            by_page.setdefault(items_page, []).append("\n".join(items))
            items.clear()

    for el in elements:
        kind = _field(el, "category") or type(el).__name__
        text = (_field(el, "text") or "").strip()
        meta = _field(el, "metadata") or {}
        page = int(_field(meta, "page_number") or 1)
        if kind != "ListItem":
            flush()
        if kind in ("Header", "Footer", "PageNumber", "PageBreak"):
            continue
        if kind == "ListItem":
            if items and page != items_page:
                flush()
            items_page = page
            if text:
                items.append(f"- {text}")
            continue
        if kind == "Title":
            depth = _field(meta, "category_depth")
            out = f"{'#' * (min(int(depth or 0), 5) + 1)} {text}" if text else ""
        elif kind == "Table":
            out = _field(meta, "text_as_html") or text
        elif kind in ("Image", "Figure"):
            out = "[image]"
        else:  # NarrativeText, FigureCaption, Formula, Address, UncategorizedText...
            out = text
        if out:
            by_page.setdefault(page, []).append(out)
    flush()
    if not by_page:
        return []
    return ["\n\n".join(by_page.get(i, [])) for i in range(1, max(by_page) + 1)]


def unstructured_pages(parser, path: Path) -> list[str]:
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=str(path), strategy=parser.option("strategy", "hi_res"),
                             infer_table_structure=True)
    pages = Pages(unstructured_elements_to_pages(elements))
    pages.raw = [e.to_dict() for e in elements if hasattr(e, "to_dict")]
    return pages


entry("unstructured", "Unstructured (partition_pdf)", "library", unstructured_pages, requires="unstructured",
       install='uv add "unstructured[pdf]"   (hi_res also needs poppler and tesseract on the PATH)',
       options={"strategy": "hi_res"}, version="latest release (verify)", license="Apache-2.0",
       source="https://github.com/Unstructured-IO/unstructured",
       note="Heavy install. strategy: hi_res (layout model, tables), fast (text layer only) or ocr_only. Titles "
            "become headings by category_depth, tables come as HTML, there is no footnote element. On Windows, "
            "poppler and tesseract are the usual stumbling block: WSL is easier")
