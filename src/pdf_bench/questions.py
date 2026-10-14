"""Each row of a score is a *question* put to the output: "is this element there, as this kind of element, in
the right place?". The deterministic scorer answers with rules; the LLM judge (``judge.py``) answers the same
questions in words. This module writes the question for a ground-truth element, so both read the same thing.
"""

from __future__ import annotations

from .elements import Element

# What "content" and "structure" mean for each kind, as the judge is told. The scorer's rules are the reference
# these sentences describe; the judge never sees the rules themselves, so it is a second opinion, not a copy.
RUBRIC: dict[str, str] = {
    "heading": "content = the text is present, even as plain text. structure = 1 if the text is a heading at that level in the output (Markdown '#' count, HTML <hN>), else 0. If at least three headings on two levels "
               "are all shifted by the same amount (the tool numbers its first level as H2), the hierarchy is right: count 1. A single heading at the wrong level is wrong.",
    "paragraph": "structure = 1 if the text forms ONE paragraph block of its own: not merged with neighbouring text, not cut in pieces, not a heading or list item; else 0.",
    "list": "structure = the fraction of items that are list items at the right nesting depth (bullet or number style does not matter; "
            "lines that start with a bullet glyph such as • count as list items).",
    "table": "content = the fraction of the cell texts found anywhere in the output (tick marks and positions are structure, not content). structure = the fraction of the counted cells that are in the right row and column of a real table (merged cells keep their span, "
             "nested tables are tables). Cells of a nested table that were flattened into the outer table count half.",
    "link": "content = the visible text is present (the target is structure, not content). structure = 1 if a link whose target is that URL covers the visible text (it may cover a few more words), else 0.",
    "footnote": "structure = 1 if the note is marked as a footnote or written as its own block starting with its number, else 0.",
    "figure": "structure = 1 if a figure (image marker or description) sits next to the caption, or the chart is returned as data "
              "(most of its labels and values just before the caption), else 0. The text inside the image is NOT needed for content.",
    "caption": "structure = 1 if the caption sits next to a figure or a table, else 0.",
    "field": "structure = 1 if the value follows its label, before the next field's label, else 0. A blank field has no value: "
             "only its label can be checked (structure = 1 if the label is in the output).",
    "checkbox": "structure = 1 if the box is shown with the right state (ticked or empty) next to its label, else 0. Any clear mark counts "
                "(☒ ✓ ✗ [x] \\times for ticked; ☐ [ ] \\square for empty).",
    "header": "structure = 1 if the running header is NOT in the output (it must be removed), else 0. content has no meaning here.",
    "footer": "structure = 1 if the running footer is NOT in the output (it must be removed), else 0. content has no meaning here.",
    "reading_order": "structure = the fraction of consecutive pairs of paragraphs that come in the expected order.",
    "table_continuity": "structure = 1 if the table that spans several pages is ONE table in the output, else 0.",
    "numbers": "content = the fraction of the listed numeric values found with exactly the same digits. structure has no meaning here.",
    "title": "structure = 1 if the title is the first heading at level 1 (or at the level the document's headings are all shifted to, when at least three headings show it); 0.5 if it is the "
             "first heading at another level; else 0.",
    "section_path": "structure = 1 if the block sits under exactly that chain of headings in the output, else 0. Compare the hierarchy of the "
                    "headings, not their absolute levels. The block counts wherever its text is, even if the tool did not mark it as that kind of block.",
    "page": "structure = the fraction of elements found on the right page (output pages are separated by page markers).",
}


def _cells(table: Element, indent: str = "") -> list[str]:
    out = []
    for c in sorted(table.children, key=lambda c: (c.attrs.get("r", 0), c.attrs.get("c", 0))):
        bits = [f"row {c.attrs.get('r', 0) + 1}, col {c.attrs.get('c', 0) + 1}"]
        if c.attrs.get("rowspan", 1) > 1:
            bits.append(f"spans {c.attrs['rowspan']} rows")
        if c.attrs.get("colspan", 1) > 1:
            bits.append(f"spans {c.attrs['colspan']} columns")
        if c.attrs.get("header"):
            bits.append("header cell")
        text = c.text or ", ".join(f"checkbox {'ticked' if k.attrs.get('checked') else 'empty'}: {k.text}"
                                   for k in c.children if k.type == "checkbox")
        out.append(f"{indent}- ({'; '.join(bits)}) {text!r}")
        for n in c.children:
            if n.type == "table":
                out.append(f"{indent}  nested table inside that cell:")
                out += _cells(n, indent + "    ")
    return out


def _items(lst: Element, depth: int = 0) -> list[str]:
    out = []
    for i, it in enumerate(lst.children, 1):
        mark = f"{i}." if lst.attrs.get("ordered") else "-"
        out.append("  " * depth + f"{mark} {it.text}")
        for k in it.children:
            if k.type == "list":
                out += _items(k, depth + 1)
    return out


def spec(e: Element) -> str:
    """The expected element in words, with everything the judge needs to decide (and nothing about the output)."""
    t = e.type
    if t == "heading":
        return f"A heading of level {e.level} (H{e.level}) with the text {e.text!r}."
    if t == "paragraph":
        return f"A paragraph with the text {e.text!r}."
    if t == "list":
        kind = "numbered" if e.attrs.get("ordered") else "bulleted"
        return f"A {kind} list with these items (indentation = nesting depth):\n" + "\n".join(_items(e))
    if t == "table":
        rows, cols = e.attrs.get("rows"), e.attrs.get("cols")
        extra = " It spans several pages." if e.attrs.get("kind") == "multipage" else ""
        return f"A table of {rows} rows and {cols} columns.{extra} Its cells:\n" + "\n".join(_cells(e))
    if t == "link":
        return f"A link whose visible text is {e.text!r} and whose target is {e.attrs.get('href')!r}."
    if t == "footnote":
        return f"A footnote with marker {e.attrs.get('marker')!r} and the text {e.text!r}."
    if t == "figure":
        cap = next((c.text for c in e.children if c.type == "caption"), "")
        words = e.attrs.get("image_text")
        return f"A figure (an image) with the caption {cap!r}." + (f" The image itself contains the text: {words}." if words else "")
    if t == "caption":
        return f"A caption with the text {e.text!r}."
    if t == "field":
        v = e.attrs.get("value", "")
        return f"A form field with the label {e.attrs.get('label')!r} and the value {v!r}." if v else \
            f"A blank form field with the label {e.attrs.get('label')!r}."
    if t == "checkbox":
        return f"A checkbox, {'ticked' if e.attrs.get('checked') else 'empty'}, with the label {e.text!r}."
    if t in ("header", "footer"):
        return f"The running {t} {e.text!r}, repeated on every page: it should not be part of the body text."
    return f"A {t}: {e.text!r}"
