"""Find on which page of the PDF each asserted element sits, from the PDF's own text layer.

The ground truth does not store pages (they depend on the layout); they are read back from the PDF, so datasets made
before this existed work too. Only elements whose text is found on exactly one page are given a page; a PDF without a
text layer (a scan) gets none.
"""

from __future__ import annotations

from pathlib import Path

from .elements import Element, GroundTruth

KINDS = ("heading", "paragraph", "caption", "footnote")


def _key(e: Element) -> str:
    from .score import norm

    text = e.text
    if e.type == "figure":
        text = next((c.text for c in e.children if c.type == "caption"), "")
    return norm(text)[:60]


def annotate(gt: GroundTruth, pdf: str | Path) -> GroundTruth:
    """A copy of ``gt`` whose headings, paragraphs, captions and footnotes carry ``attrs["page"]`` (1-based)."""
    import copy

    import pymupdf

    from .score import norm

    out = copy.deepcopy(gt)
    with pymupdf.open(pdf) as doc:
        texts = [norm(p.get_text()) for p in doc]
    if not any(texts):
        return out
    for e in out.elements:
        kind = "caption" if e.type == "figure" else e.type
        if kind not in KINDS:
            continue
        key = _key(e)
        if len(key) < 8:
            continue
        where = [i + 1 for i, t in enumerate(texts) if key in t]
        if len(where) == 1:
            e.attrs["page"] = where[0]
    return out
