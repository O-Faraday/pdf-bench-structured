"""Write the ground truth back out as HTML or Markdown.

Two uses: a check that the scorer gives full marks to a perfect answer (tests), and two reference
"parsers" in the app (``oracle_html``, ``oracle_markdown``) that read the ground truth next to a PDF.
They show the ceiling of each output format: HTML can carry merged and nested cells, Markdown cannot.
They are not parsers: they never look at the PDF.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from .elements import Element, GroundTruth, load_gt

_SKIP = ("header", "footer")


def _gt_for(pdf: Path) -> GroundTruth:
    path = pdf.with_suffix(".gt.json")
    if not path.is_file():
        raise FileNotFoundError(f"no ground truth next to the PDF: {path.name}")
    return load_gt(path)


# --- HTML ------------------------------------------------------------------------------------------------

def _html_table(t: Element) -> str:
    rows: dict[int, list[Element]] = {}
    for c in t.children:
        rows.setdefault(c.attrs["r"], []).append(c)
    out = ["<table>"]
    for r in sorted(rows):
        out.append("<tr>")
        for c in sorted(rows[r], key=lambda x: x.attrs["c"]):
            tag = "th" if c.attrs.get("header") else "td"
            span = "".join(f' {k}="{c.attrs[k]}"' for k in ("rowspan", "colspan") if k in c.attrs)
            inner = escape(c.text) + "".join(_html_table(n) for n in c.children if n.type == "table")
            if any(n.type == "checkbox" for n in c.children):
                box = next(n for n in c.children if n.type == "checkbox")
                inner = "☒" if box.attrs.get("checked") else "☐"
            out.append(f"<{tag}{span}>{inner}</{tag}>")
        out.append("</tr>")
    return "".join(out) + "</table>"


def _html_list(lst: Element) -> str:
    tag = "ol" if lst.attrs.get("ordered") else "ul"
    items = "".join(f"<li>{escape(i.text)}{''.join(_html_list(k) for k in i.children)}</li>" for i in lst.children)
    return f"<{tag}>{items}</{tag}>"


def gt_to_html(gt: GroundTruth) -> str:
    out = []
    for e in gt.elements:
        if e.type in _SKIP:
            continue
        if e.type == "heading":
            out.append(f"<h{e.level}>{escape(e.text)}</h{e.level}>")
        elif e.type in ("paragraph", "caption"):
            out.append(f"<p>{escape(e.text)}</p>")
        elif e.type == "list":
            out.append(_html_list(e))
        elif e.type == "link":
            out.append(f'<p><a href="{escape(e.attrs["href"])}">{escape(e.text)}</a></p>')
        elif e.type == "footnote":
            out.append(f'<aside data-footnote="{escape(e.attrs["marker"])}">{escape(e.text)}</aside>')
        elif e.type == "table":
            out.append(_html_table(e))
        elif e.type == "field":
            out.append(f"<p>{escape(e.attrs['label'])}: {escape(e.attrs.get('value', ''))}</p>")
        elif e.type == "checkbox":
            out.append(f"<p>{'☒' if e.attrs.get('checked') else '☐'} {escape(e.text)}</p>")
        elif e.type == "figure":
            cap = "".join(f"<figcaption>{escape(c.text)}</figcaption>" for c in e.children if c.type == "caption")
            out.append(f'<figure><img alt="">{cap}</figure>')
    return "\n\n".join(out)


# --- Markdown ----------------------------------------------------------------------------------------------

def _flat(c: Element) -> str:
    parts = [c.text] if c.text else []
    for n in c.children:
        if n.type == "table":
            parts += [x for cell in n.children for x in [_flat(cell)] if x]
        elif n.type == "checkbox":
            parts.append("☒" if n.attrs.get("checked") else "☐")
    return " ; ".join(parts).replace("|", "\\|")


def _md_table(t: Element) -> str:
    n_rows, n_cols = t.attrs["rows"], t.attrs["cols"]
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for c in t.children:
        grid[c.attrs["r"]][c.attrs["c"]] = _flat(c)
    lines = ["| " + " | ".join(grid[0]) + " |", "|" + " --- |" * n_cols]
    lines += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(lines)


def _md_list(lst: Element, depth: int = 0) -> str:
    lines = []
    for k, item in enumerate(lst.children, 1):
        bullet = f"{k}." if lst.attrs.get("ordered") else "-"
        lines.append("   " * depth + f"{bullet} {item.text}")
        lines += [_md_list(n, depth + 1) for n in item.children if n.type == "list"]
    return "\n".join(lines)


def gt_to_markdown(gt: GroundTruth) -> str:
    out = []
    for e in gt.elements:
        if e.type in _SKIP:
            continue
        if e.type == "heading":
            out.append("#" * e.level + " " + e.text)
        elif e.type == "paragraph":
            out.append(e.text)
        elif e.type == "caption":
            out.append(f"*{e.text}*")
        elif e.type == "list":
            out.append(_md_list(e))
        elif e.type == "link":
            out.append(f"[{e.text}]({e.attrs['href']})")
        elif e.type == "footnote":
            out.append(f"[^{e.attrs['marker']}]: {e.text}")
        elif e.type == "table":
            out.append(_md_table(e))
        elif e.type == "field":
            out.append(f"**{e.attrs['label']}**: {e.attrs.get('value', '')}")
        elif e.type == "checkbox":
            out.append(f"{'☒' if e.attrs.get('checked') else '☐'} {e.text}")
        elif e.type == "figure":
            out.append("![](image.png)")
            out += [f"*{c.text}*" for c in e.children if c.type == "caption"]
    return "\n\n".join(out)


def oracle_html_pages(parser, path: Path) -> list[str]:
    return [gt_to_html(_gt_for(path))]


def oracle_markdown_pages(parser, path: Path) -> list[str]:
    return [gt_to_markdown(_gt_for(path))]
