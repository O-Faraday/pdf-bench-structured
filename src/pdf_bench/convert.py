"""Parser output (Markdown, with or without HTML) -> the element model of :mod:`pdf_bench.elements`.

This is what makes parsers comparable whatever their output looks like: the score never looks at
syntax, only at the elements recovered.

Recognised: headings (``#``..``######``, ``<h1>``..), paragraphs, bullet and numbered lists with nesting,
GitHub tables and HTML tables (``rowspan``, ``colspan``, ``<th>``, nested tables), links (``[t](u)``,
``<a href>``), figures (``![..](..)``, ``[image]``, ``<!-- image -->``, ``<img>``, ``<figure>`` with
``<figcaption>``), footnotes (``[^1]: text`` lines, ``<aside data-footnote="1">``).
Form fields and checkboxes are not converted here: the scorer reads them from the text (see
:mod:`pdf_bench.score`).
"""

from __future__ import annotations

import html as _html
import re

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from markdown_it import MarkdownIt

from .elements import Element

_MD = MarkdownIt("commonmark", {"html": True}).enable("table")
_FOOTNOTE = re.compile(r"^\[\^([^\]\s]+)\]:\s*(.+?)\s*$", re.M)
_FIGURE_TEXT = re.compile(r"^\s*(?:!?\[(?:image|figure|picture|img|chart|graph|diagram|illustration|photo)(?:\s*:[^\]]*)?\]|<!--\s*image\s*-->)\s*$", re.I)
# PyMuPDF4LLM writes the text of a picture between these comments and nothing else marks the picture
_PICTURE = re.compile(r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->", re.S | re.I)
_HEADINGS = {f"h{i}": i for i in range(1, 7)}


def to_elements(text: str | list[str]) -> list[Element]:
    """Elements of a parser output, in reading order. A list of pages is joined with blank lines."""
    if isinstance(text, list):
        text = "\n\n".join(text)
    text = _tight_tables(_PICTURE.sub("\n\n[image]\n\n", text))
    notes = [Element("footnote", m.group(2), attrs={"marker": m.group(1)}) for m in _FOOTNOTE.finditer(text)]
    text = _FOOTNOTE.sub("", text)
    tokens = _MD.parse(text)
    return _blocks(tokens, 0, len(tokens)) + notes


_TABLE_TAG = re.compile(r"<(/?)table\b[^>]*>", re.I)


def _tight_tables(text: str) -> str:
    """Remove the blank lines inside an HTML ``<table>`` (nested ones included): a blank line ends an HTML block in
    Markdown, which cut a table that holds nested tables in several pieces and lost every nested cell."""
    out, depth, start, last = [], 0, 0, 0
    for m in _TABLE_TAG.finditer(text):
        if m.group(1) == "":
            if depth == 0:
                start = m.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                out.append(text[last:start])
                out.append(re.sub(r"\n[ \t]*(?:\n[ \t]*)+", "\n", text[start: m.end()]))
                last = m.end()
    out.append(text[last:])
    return "".join(out)


# --- markdown tokens ------------------------------------------------------------------------------------

def _end(tokens, i: int) -> int:
    level = 0
    for j in range(i, len(tokens)):
        level += tokens[j].nesting
        if level == 0:
            return j
    return len(tokens) - 1


def _plain(inline) -> str:
    out = []
    for c in inline.children or []:
        if c.type in ("text", "code_inline"):
            out.append(c.content)
        elif c.type in ("softbreak", "hardbreak"):
            out.append(" ")
        elif c.type == "html_inline" and re.match(r"<br\s*/?>", c.content, re.I):
            out.append(" ")
    return " ".join("".join(out).split())


def _links(inline) -> list[Element]:
    out, kids, href = [], None, None
    for c in inline.children or []:
        if c.type == "link_open":
            href, kids = c.attrGet("href") or "", []
        elif c.type == "link_close" and kids is not None:
            out.append(Element("link", " ".join("".join(kids).split()), attrs={"href": href}))
            kids = None
        elif kids is not None and c.type in ("text", "code_inline"):
            kids.append(c.content)
    return out


def _images(inline) -> list[Element]:
    return [Element("figure", attrs={"alt": c.content or ""}) for c in inline.children or [] if c.type == "image"]


def _blocks(tokens, lo: int, hi: int) -> list[Element]:
    out: list[Element] = []
    i = lo
    while i < hi:
        t = tokens[i]
        if t.type == "heading_open":
            j = _end(tokens, i)
            out.append(Element("heading", _plain(tokens[i + 1]), level=int(t.tag[1]), children=_links(tokens[i + 1])))
            i = j + 1
        elif t.type == "paragraph_open":
            j = _end(tokens, i)
            out += _paragraph(tokens[i + 1])
            i = j + 1
        elif t.type in ("bullet_list_open", "ordered_list_open"):
            j = _end(tokens, i)
            out.append(_list(tokens, i, j))
            i = j + 1
        elif t.type == "table_open":
            j = _end(tokens, i)
            out.append(_gfm_table(tokens, i, j))
            i = j + 1
        elif t.type == "blockquote_open":
            j = _end(tokens, i)
            out += _blocks(tokens, i + 1, j)
            i = j + 1
        elif t.type == "html_block":
            out += _html_blocks(t.content)
            i += 1
        elif t.type in ("fence", "code_block"):
            if t.content.strip():
                out.append(Element("paragraph", " ".join(t.content.split())))
            i += 1
        else:
            i += 1
    return out


def _paragraph(inline) -> list[Element]:
    text = _plain(inline)
    figures = _images(inline)
    if not text and figures:
        return figures
    if _FIGURE_TEXT.match(text):
        return [Element("figure")]
    # a line "[image]" inside a longer paragraph (soft-wrapped) is still a figure
    parts = [p for p in re.split(r"\s*\[image\]\s*", text, flags=re.I)] if "[image]" in text.lower() else [text]
    out: list[Element] = []
    for k, part in enumerate(parts):
        if part:
            out.append(Element("paragraph", part, children=_links(inline) if k == 0 else []))
        if k < len(parts) - 1:
            out.append(Element("figure"))
    return out + figures


def _list(tokens, i: int, j: int) -> Element:
    ordered = tokens[i].type == "ordered_list_open"
    items, k = [], i + 1
    while k < j:
        if tokens[k].type == "list_item_open":
            e = _end(tokens, k)
            inner = _blocks(tokens, k + 1, e)
            text = next((b.text for b in inner if b.type == "paragraph"), "")
            kids = [b for b in inner if b.type in ("list", "table")]
            items.append(Element("list_item", text, children=kids))
            k = e + 1
        else:
            k += 1
    return Element("list", attrs={"ordered": ordered}, children=items)


def _gfm_table(tokens, i: int, j: int) -> Element:
    cells, r, c, head = [], -1, 0, False
    for k in range(i, j):
        t = tokens[k]
        if t.type == "thead_open":
            head = True
        elif t.type == "tbody_open":
            head = False
        elif t.type == "tr_open":
            r, c = r + 1, 0
        elif t.type in ("th_open", "td_open"):
            cells.append(Element("cell", _plain(tokens[k + 1]), attrs={"r": r, "c": c, "header": head}))
            c += 1
    return Element("table", attrs={"rows": r + 1, "cols": max((x.attrs["c"] for x in cells), default=-1) + 1,
                                   "syntax": "markdown"}, children=cells)


# --- html -------------------------------------------------------------------------------------------------

def _text(tag: Tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split())


def _a_links(tag: Tag) -> list[Element]:
    return [Element("link", _text(a), attrs={"href": a.get("href", "")}) for a in tag.find_all("a") if a.get("href")]


_BLOCK_TAGS = ["table", "p", "ul", "ol", "div", "section", "article", "figure", "aside", "h1", "h2", "h3", "h4", "h5", "h6"]


def _html_blocks(content: str) -> list[Element]:
    soup = BeautifulSoup(content, "html.parser")
    out: list[Element] = []
    for node in soup.children:
        if isinstance(node, Comment):
            if _FIGURE_TEXT.match(f"<!--{node}-->"):
                out.append(Element("figure"))
        elif isinstance(node, Tag):
            out += _html_node(node)
        elif isinstance(node, NavigableString) and str(node).strip():
            out.append(Element("paragraph", " ".join(str(node).split())))
    return out


def _html_node(tag: Tag) -> list[Element]:
    name = tag.name
    if name == "table":
        return [_html_table(tag)]
    if name in _HEADINGS:
        return [Element("heading", _text(tag), level=_HEADINGS[name], children=_a_links(tag))]
    if name in ("ul", "ol"):
        return [_html_list(tag)]
    if name == "img":
        return [Element("figure", attrs={"alt": tag.get("alt", "")})]
    if name == "figure":
        cap = tag.find("figcaption")
        fig = Element("figure", children=[Element("caption", _text(cap))] if cap else [])
        return [fig]
    if name == "aside" and tag.get("data-footnote") is not None:
        return [Element("footnote", _text(tag), attrs={"marker": tag.get("data-footnote")})]
    if name in ("div", "section", "article", "body", "html", "main", "center"):
        if not tag.find(_BLOCK_TAGS):  # <div style="text-align: center;">Figure 1: ...</div>: just a paragraph
            text = _text(tag)
            if _FIGURE_TEXT.match(text):
                return [Element("figure")]
            return [Element("paragraph", text, children=_a_links(tag))] if text else []
        out: list[Element] = []
        for child in tag.children:
            if isinstance(child, Tag):
                out += _html_node(child)
            elif isinstance(child, NavigableString) and not isinstance(child, Comment) and str(child).strip():
                out.append(Element("paragraph", " ".join(str(child).split())))
        return out
    if name == "a" and tag.get("href"):
        return [Element("link", _text(tag), attrs={"href": tag["href"]})]
    text = _text(tag)
    if _FIGURE_TEXT.match(text):
        return [Element("figure")]
    return [Element("paragraph", text, children=_a_links(tag))] if text else []


def _html_list(tag: Tag) -> Element:
    items = []
    for li in tag.find_all("li", recursive=False):
        kids = [_html_list(sub) for sub in li.find_all(["ul", "ol"], recursive=False)]
        own = BeautifulSoup(str(li), "html.parser").li
        for sub in own.find_all(["ul", "ol"]):
            sub.decompose()
        items.append(Element("list_item", _text(own), children=kids))
    return Element("list", attrs={"ordered": tag.name == "ol"}, children=items)


def _rows(table: Tag):
    for child in table.children:
        if isinstance(child, Tag):
            if child.name == "tr":
                yield child, child.parent.name == "thead"
            elif child.name in ("thead", "tbody", "tfoot"):
                for tr in child.find_all("tr", recursive=False):
                    yield tr, child.name == "thead"


def _html_table(table: Tag) -> Element:
    cells: list[Element] = []
    taken: set[tuple[int, int]] = set()
    n_rows = n_cols = 0
    for r, (tr, in_head) in enumerate(_rows(table)):
        c = 0
        for td in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in taken:
                c += 1
            rs = max(1, _int(td.get("rowspan")))
            cs = max(1, _int(td.get("colspan")))
            for dr in range(rs):
                for dc in range(cs):
                    taken.add((r + dr, c + dc))
            nested = [t for t in td.find_all("table") if t.find_parent("table") is table]
            own = BeautifulSoup(str(td), "html.parser").find(["td", "th"])
            for t in own.find_all("table"):
                t.decompose()
            attrs = {"r": r, "c": c, "header": td.name == "th" or in_head}
            if rs > 1:
                attrs["rowspan"] = rs
            if cs > 1:
                attrs["colspan"] = cs
            cells.append(Element("cell", _text(own), attrs=attrs, children=[_html_table(t) for t in nested]))
            c += cs
            n_cols = max(n_cols, c)
        n_rows = r + 1
    return Element("table", attrs={"rows": n_rows, "cols": n_cols, "syntax": "html"}, children=cells)


def _int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def html_unescape(text: str) -> str:
    return _html.unescape(text)
