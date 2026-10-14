"""Compare a parser's output with the ground truth, element by element.

Every element of the ground truth gets two scores between 0 and 1:

* **content**: is the text there at all, wherever it is? Independent of the output format;
* **structure**: was it recovered *as that kind of element*, correctly placed? (a heading at the right
  level, a table cell at the right row and column with its spans, a list item at the right depth, a link
  with its target, a ticked box shown as ticked...).

A parser that returns flat text gets content without structure: that gap is what this benchmark measures.
Text is compared after removing Markdown/HTML noise and case; a match is a similarity of at least 90%
(short strings must match exactly). Scores of tables and lists are fractions (cells or items right).

Heuristics to know about: form fields and checkboxes are read from the text (a value must follow its
label closely; a ticked state must be written with a recognisable mark: ☒ ☑ ✓ [x]); footnotes count as
structure when the output marks them (``[^1]: text``, ``<aside data-footnote>``) or writes them as their own block starting
with their number (``1 text``); running headers
and footers score structure 1 when they are absent from the output.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
from rapidfuzz import fuzz

from . import questions
from .convert import html_unescape, to_elements
from .elements import Element, GroundTruth

THRESHOLD = 90
CHART_BLOCK_CHARS = 100
CHECKED = ("☒", "☑", "✓", "✔", "✅", "✗", "✘", "☓", "[x]", "[X]", "(x)", "\\times", "\\checkmark", "\\boxtimes")  # \times: LaTeX from OCR models
UNCHECKED = ("☐", "□", "[ ]", "[]", "( )", "\\square", "\\Box")


def norm(text: str) -> str:
    """Comparison form: no tags, no Markdown emphasis or escapes, one space, lower case."""
    t = html_unescape(re.sub(r"<[^>]+>", " ", text or ""))
    t = t.replace("\\", "").replace("*", "").replace("`", "").replace("_", " ")
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("\xa0", " ")
    return " ".join(t.split()).lower()


SHORT = 12  # characters: below this a text must be there as whole words, not approximately ("Signatures" is not "[signature]")


def found(needle: str, hay: str) -> bool:
    n = norm(needle)
    if not n:
        return True
    if len(n) <= SHORT:
        return re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", hay) is not None
    return n in hay or fuzz.partial_ratio(n, hay) >= THRESHOLD


def same(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    return a == b or (len(a) >= 5 and len(b) >= 5 and fuzz.ratio(a, b) >= THRESHOLD)


def covers(expected: str, candidate: str, share: float = 0.8) -> bool:
    """The candidate holds the expected text and is not just a fragment of it.

    ``fuzz.partial_ratio`` takes the shorter text as the needle, so a fragment of the expected text, or a
    short block contained in it ("Claims" in a caption), matches at 100%: the candidate must also be at
    least ``share`` as long as the expected text.
    """
    g, c = norm(expected), norm(candidate)
    return bool(g) and len(c) >= share * len(g) and fuzz.partial_ratio(g, c) >= THRESHOLD


def chunk_coverage(text: str, hay: str, words: int = 6) -> float:
    """Share of the text, cut in runs of ``words`` words, that is found in ``hay``: a paragraph cut in two blocks (the
    tail written after the next section, as two-column pages do) is still in the output, which one contiguous search misses."""
    ws = norm(text).split()
    chunks = [" ".join(ws[i: i + words]) for i in range(0, len(ws), words)]
    chunks = [c for c in chunks if len(c) >= 12] or chunks
    return sum(c in hay or fuzz.partial_ratio(c, hay) >= THRESHOLD for c in chunks) / len(chunks) if chunks else 0.0


def level_offset(gt_headings: list[Element], flat: list[Element]) -> int:
    """The constant by which a tool shifts every heading level (Docling writes its first level as ``##``), else 0.

    Only when at least three headings and two different levels were found, and every one is shifted by the same amount:
    then the hierarchy is right and only its numbering differs, which is what chunking needs.
    """
    diffs, levels = set(), set()
    for e in gt_headings:
        cand = next((p for p in flat if p.type == "heading" and same(e.text, p.text)), None)
        if cand is not None and cand.level and e.level:
            diffs.add(cand.level - e.level)
            levels.add(e.level)
    return diffs.pop() if len(diffs) == 1 and len(levels) >= 2 and sum(
        any(p.type == "heading" and same(e.text, p.text) for p in flat) for e in gt_headings) >= 3 else 0


def box_state(text: str) -> bool | None:
    """True/False for a recognisable ticked/empty mark in ``text`` (the last one wins), else None."""
    best, state = -1, None
    for marks, value in ((CHECKED, True), (UNCHECKED, False)):
        for m in marks:
            i = text.rfind(m)
            if i > best:
                best, state = i, value
    if state is None and text.strip() in ("x", "X"):
        return True
    return state


_AFTER_BOX = re.compile(r"\s*:?\s*(?=[\[(\u2610-\u2612\u2713\u2714\u2705\u25a1])")


def _box_around(hay: str, m) -> bool | None:
    """The state of the box written just before the label, else (``label: [x]``) just after it."""
    before = box_state(hay[max(0, m.start() - 8): m.start()])
    if before is not None:
        return before
    after = hay[m.end(): m.end() + 12]
    return box_state(after[:len(a.group(0)) + 4]) if (a := _AFTER_BOX.match(after)) else None


_NOTE_SPLIT = re.compile(r"(?:\n|(?<=[.!?])\s+)(?=[\[\(\^<]*(?:sup>)?\d+(?:</sup>)?[\]\).:]?\s)")


def _note_lines(text: str) -> list[str]:
    """Several numbered notes written in one block ("1 First note. 2 Second note.") one by one."""
    return [x.strip() for x in _NOTE_SPLIT.split(text) if x.strip()]


def _chart_read(words: list[str], blocks: list[Element]) -> bool:
    """Most of the text drawn inside a chart (title and labels) is in the blocks around its caption: the model read the
    chart as data. A table of values with the chart's title above it or missing still counts (three words in four)."""
    small = [b for b in blocks if len(_block_text(b).strip()) <= CHART_BLOCK_CHARS]  # a chart's data is short: not a whole table of the page
    text = norm(" ".join(t for block in small for t in (
        block.text, *(c.text for c in block.children), *(g.text for c in block.children for g in c.children))))
    hits = sum(bool(re.search(r"(?<!\w)" + re.escape(norm(w)) + r"(?!\w)", text)) for w in words)
    return bool(words) and hits >= max(2, 0.75 * len(words))


@dataclass
class Row:
    kind: str
    expected: str
    content: float | None
    structure: float | None
    detail: str = ""
    spec: str = ""  # the question in words (questions.py): what the LLM judge is asked about this row


@dataclass
class Report:
    document: str
    parser: str
    rows: list[Row] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.rows], columns=["kind", "expected", "content", "structure", "detail", "spec"])

    def summary(self) -> pd.DataFrame:
        f = self.frame()
        if f.empty:
            return pd.DataFrame(columns=["elements", "content", "structure"])
        return f.groupby("kind").agg(elements=("kind", "size"), content=("content", "mean"),
                                     structure=("structure", "mean")).round(3)

    def overall(self) -> dict[str, float | None]:
        """Mean of the per-kind means, so a 200-cell table does not outweigh a single heading."""
        s = self.summary()
        return {"content": _mean(s["content"]), "structure": _mean(s["structure"])}


def _mean(series) -> float | None:
    v = series.dropna()
    return round(float(v.mean()), 3) if len(v) else None


# --- measures added after the first real run: exact numbers, title, section path, page -------------------------

def _numbers_row(gt, hay, add):
    want = _gt_numbers(gt)
    if not want:
        return
    got = Counter(_numbers(hay))
    hit = sum(min(n, got[v]) for v, n in want.items())
    total = sum(want.values())
    lost = [v for v, n in want.items() if got[v] < n]
    add("numbers", f"{total} numeric values (amounts, dates, times)", hit / total, None,
        "" if not lost else f"{len(lost)} value(s) missing or altered, e.g. {', '.join(lost[:4])}",
        spec="These numeric values (amounts, dates, times; a thousands group may be written with spaces) must appear "
             "with exactly the same digits, as often as listed: " + ", ".join(f"{v} (x{n})" if n > 1 else v for v, n in want.items()))


def _title_row(gt, flat, hay, add, shift=0):
    title = next((e for e in gt.elements if e.type == "heading"), None)
    if title is None or title.level != 1:
        return
    heads = [p for p in flat if p.type == "heading"]
    content = float(found(title.text, hay))
    if heads and same(title.text, heads[0].text):
        ok, why = ((1.0, "") if heads[0].level == 1 + shift else (0.5, f"first heading is the title but at H{heads[0].level}, not H1"))
        if shift and ok == 1.0:
            why = f"hierarchy right, every level shifted by {shift:+d}"
    elif any(same(title.text, h.text) for h in heads):
        ok, why = 0.0, "the title is a heading but not the first one"
    else:
        ok, why = 0.0, "title not detected as a heading" if content else "title missing"
    add("title", title.text, content, ok, why, spec=f"The document title {title.text!r}: the first heading of the output, at level 1 (H1).")


def _section_rows(gt, predicted, add):
    got = _with_paths(predicted)
    for block, path in _with_paths(gt.elements):
        if not path:
            continue
        key = _block_key(block)
        match = next((p for e, p in got if e.type == block.type and key and (
            covers(key, _block_key(e)) if block.type == "paragraph" else same(key, _block_key(e)))), None)
        if match is None and key:  # written as another kind of block (a list as plain lines): its context still counts
            match = next((p for e, p in got if e.type != block.type and norm(key) in norm(_block_text(e))), None)
        want = " > ".join(path)
        if match is None:
            add("section_path", f"{block.type} under {want}", None, 0.0, "block not found", spec=_section_spec(block, want))
            continue
        ok = len(match) == len(path) and all(same(a, b) for a, b in zip(path, match))
        add("section_path", f"{block.type} under {want}", None, float(ok),
            "" if ok else f"headings above it in the output: {' > '.join(match) or '(none)'}", spec=_section_spec(block, want))


def _section_spec(block: Element, want: str) -> str:
    start = block.text[:70] if block.type != "table" else _first_cell(block)
    return (f"The {block.type} that starts with {start!r} must sit under exactly this chain of headings "
            f"(outermost first): {want}.")


def _page_row(gt, pages, add):
    placed = [(e, e.attrs["page"]) for e in gt.elements if e.attrs.get("page")]
    if pages is None or len({p for _, p in placed}) < 2:
        return  # one page, or no text layer to place the elements: nothing to check
    if len(pages) < 2:
        add("page", f"{len(placed)} elements on {len({p for _, p in placed})} pages", None, 0.0, "the output is not split by page")
        return
    hays = [norm(p) for p in pages]
    wrong = []
    for e, page in placed:
        key = norm(next((c.text for c in e.children if c.type == "caption"), "") if e.type == "figure" else e.text)[:60]
        at = next((i + 1 for i, h in enumerate(hays) if key and (key in h or fuzz.partial_ratio(key, h) >= THRESHOLD)), None)
        if at != page:
            wrong.append(f"{key[:25]!r} on page {at}, expected {page}")
    add("page", f"{len(placed)} elements on {len({p for _, p in placed})} pages", None, 1 - len(wrong) / len(placed),
        "" if not wrong else "; ".join(wrong[:3]))


# --- tables ---------------------------------------------------------------------------------------------------

def _counted(cell: Element) -> bool:
    return bool(cell.text) or any(c.type == "checkbox" for c in cell.children)


def _count_cells(table: Element) -> int:
    return sum((1 if _counted(c) else 0) + sum(_count_cells(n) for n in c.children if n.type == "table")
               for c in table.children)


def _grid(table: Element) -> dict[tuple[int, int], Element]:
    at = {}
    for c in table.children:
        for dr in range(c.attrs.get("rowspan", 1)):
            for dc in range(c.attrs.get("colspan", 1)):
                at[(c.attrs["r"] + dr, c.attrs["c"] + dc)] = c
    return at


def _table_hits(gt: Element, pred: Element, shift: int = 0) -> tuple[int, int, dict]:
    """(cells right, cells counted, why) for one ground-truth table against one predicted table.

    ``shift``: the predicted table is read ``shift`` rows lower (a two-level header written as one row moves every row up).
    """
    at = _grid(pred)
    hits = total = 0
    why = {"span": 0, "nested": 0, "position": 0, "header": 0, "shift": 0}
    for c in gt.children:
        key = (c.attrs["r"], c.attrs["c"])
        pc = at.get((key[0] + shift, key[1]))
        box = next((x for x in c.children if x.type == "checkbox"), None)
        if _counted(c):
            total += 1
            if pc is None:
                why["position"] += 1
            else:
                ok = box_state(pc.text) == box.attrs.get("checked") if box else same(c.text, pc.text)
                spans = all(c.attrs.get(k, 1) == pc.attrs.get(k, 1) for k in ("rowspan", "colspan"))
                anchored = (pc.attrs["r"], pc.attrs["c"]) == (key[0] + shift, key[1]) or not spans
                if ok and spans and anchored:
                    hits += 1
                    why["header"] += int(bool(c.attrs.get("header")) != bool(pc.attrs.get("header")))
                elif ok and (pc.attrs["r"], pc.attrs["c"]) == (key[0] + shift, key[1]):
                    hits += 0.5  # the right text in the right place, its merge lost (Markdown cannot express it): half
                    why["span"] += 1
                elif ok:
                    why["span"] += 1
                else:
                    why["position"] += 1
        for n in (x for x in c.children if x.type == "table"):
            cands = [t for t in (pc.children if pc else []) if t.type == "table"]
            if cands:
                best = max((_table_hits(n, t) for t in cands), key=lambda r: r[0])
                hits, total = hits + best[0], total + best[1]
            else:
                why["nested"] += 1
                total += _count_cells(n)
                if pred is not None:  # flattened into the table (rowspans...): the values are there, the nesting is lost: half
                    flat_text = norm(" ".join(c2.text for c2 in pred.walk()))
                    inner = _gt_texts(n)
                    hits += 0.5 * sum(found(x, flat_text) for x in inner)
    return hits, total, why


def _best_hits(gt: Element, pred: Element) -> tuple[float, int, dict]:
    """``_table_hits``, and when the table is mostly wrong, the best reading with every row shifted by the same amount
    (merged header rows) at half credit: the values are in the right columns and in the right order, only the row numbers differ."""
    hits, total, why = _table_hits(gt, pred)
    if total and hits < 0.5 * total:
        for shift in (-2, -1, 1, 2):
            h, t, w = _table_hits(gt, pred, shift)
            if 0.5 * h > hits:
                hits, why = 0.5 * h, {**w, "shift": shift}
    return hits, total, why


def _stitch(gt: Element, pieces: list[Element]) -> tuple[Element, int]:
    """One predicted table from the pieces of a table that runs over several pages.

    Each piece restarts its rows at 0 (and repeats the header), so its rows are re-numbered by finding the
    ground-truth row with the same first cell. A row already taken (a repeated header) is dropped.
    Returns the stitched table and the number of pieces that contributed.
    """
    first = {c.attrs["r"]: c.text for c in gt.children if c.attrs["c"] == 0 and c.text}
    taken: set[int] = set()
    cells: list[Element] = []
    used = 0
    for piece in pieces:
        rows: dict[int, list[Element]] = {}
        for c in piece.children:
            rows.setdefault(c.attrs["r"], []).append(c)
        gained = False
        for r, row in sorted(rows.items()):
            lead = next((c for c in row if c.attrs["c"] == 0), None)
            target = next((g for g, t in first.items() if lead is not None and same(t, lead.text)), None)
            if target is None or target in taken:
                continue
            taken.add(target)
            gained = True
            for c in row:
                cells.append(Element(c.type, c.text, c.level, {**c.attrs, "r": target}, c.children))
        used += gained
    return Element("table", attrs=dict(gt.attrs), children=cells), used


def _all_tables(elements: list[Element]) -> list[Element]:
    return [e for r in elements for e in r.walk() if e.type == "table"]


def _gt_texts(table: Element) -> list[str]:
    out = []
    for c in table.children:
        if c.text:
            out.append(c.text)
        for n in c.children:
            if n.type == "table":
                out += _gt_texts(n)
    return out


# --- the scorer -------------------------------------------------------------------------------------------------

def _items(lst: Element, depth: int = 0):
    for it in lst.children:
        yield it.text, depth, bool(lst.attrs.get("ordered"))
        for k in it.children:
            if k.type == "list":
                yield from _items(k, depth + 1)


def _context(hay: str, label: str):
    pat = re.compile(r"(?<!\w)" + re.escape(norm(label)) + r"(?!\w)")
    return list(pat.finditer(hay))


_NUMBER = re.compile(r"(?<!\d)\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?(?!\d)|\d+(?:[.,/:]\d+)*")


def _numbers(text: str) -> list[str]:
    """Numeric tokens of a text; a thousands group written with spaces ("12 480,50") is one token."""
    return [re.sub(r"[ \u00a0\u202f]", "", t) for t in _NUMBER.findall(text)]


def _gt_numbers(gt: GroundTruth) -> Counter:
    """The numeric values (amounts, dates, times, identifiers' digits) of the ground truth, as a multiset."""
    out: Counter = Counter()

    def visit(e: Element):
        if e.type in ("header", "footer"):
            return  # they must be gone from the output
        for t in (e.text, e.attrs.get("label"), e.attrs.get("value")):
            if t:
                out.update(_numbers(str(t)))
        for c in e.children:
            visit(c)

    for e in gt.elements:
        visit(e)
    return out


def _first_cell(table: Element) -> str:
    return next((c.text for c in sorted(table.children, key=lambda c: (c.attrs.get("r", 0), c.attrs.get("c", 0))) if c.text), "")


def _block_key(e: Element) -> str:
    if e.type == "list":
        return next((i.text for i in e.children if i.text), "")
    return _first_cell(e) if e.type == "table" else e.text


def _block_text(e: Element) -> str:
    return " ".join([e.text, *(c.text for c in e.children), *(g.text for c in e.children for g in c.children)])


def _with_paths(elements: list[Element]) -> list[tuple[Element, list[str]]]:
    """(block, the headings above it, outermost first) for the paragraphs, lists and tables, in reading order."""
    stack: list[tuple[int, str]] = []
    out = []
    for e in elements:
        if e.type == "heading":
            while stack and stack[-1][0] >= (e.level or 1):
                stack.pop()
            stack.append((e.level or 1, e.text))
        elif e.type in ("paragraph", "list", "table"):
            out.append((e, [t for _, t in stack]))
    return out


def score(gt: GroundTruth, output: str | list[str], *, parser: str = "", raw_pages: list[str] | None = None) -> Report:
    """Score ``output`` (the parser's pages or text) against ``gt``.

    ``raw_pages`` are the pages as the tool gave them, when ``output`` was post-processed (which may merge pages):
    they are what the page-of-each-element score reads. Elements need ``attrs["page"]`` (see :mod:`pdf_bench.locate`).
    """
    pages = raw_pages if raw_pages is not None else (output if isinstance(output, list) else None)
    text = "\n\n".join(output) if isinstance(output, list) else output
    hay = norm(text)
    predicted = to_elements(text)
    flat = [e for r in predicted for e in r.walk()]
    report = Report(gt.name, parser)
    used_tables: set[int] = set()
    tables = _all_tables(predicted)
    shift = level_offset([e for e in gt.elements if e.type == "heading"], flat)
    field_labels = [norm(x.attrs["label"]) for x in gt.elements if x.type == "field"]

    current: list[Element] = []

    def add(kind, expected, content, structure, detail="", spec=None):
        if spec is None:  # a row about the element being scored: its question is that element in words
            spec = questions.spec(current[0]) if current else ""
        report.rows.append(Row(kind, expected[:80], content, structure, detail, spec))

    for e in gt.elements:
        current[:] = [e]
        t = e.type
        if t == "heading":
            content = float(found(e.text, hay))
            cands = [p for p in flat if p.type == "heading" and same(e.text, p.text)]
            ok = any(p.level == e.level + shift for p in cands)
            detail = (("" if not shift else f"hierarchy right, every level shifted by {shift:+d}") if ok else
                      f"detected as H{cands[0].level}, expected H{e.level}" if cands else
                      "text present but not a heading" if content else "missing")
            add(t, f"H{e.level} {e.text}", content, float(ok), detail)
        elif t == "paragraph":
            content = float(found(e.text, hay))
            if not content:
                content = chunk_coverage(e.text, hay)
            g = norm(e.text)
            blocks = [p for p in flat if p.type == "paragraph"]
            tight = any(covers(e.text, p.text) and len(norm(p.text)) <= 1.3 * len(g) + 20 for p in blocks)
            loose = any(covers(e.text, p.text) for p in blocks)
            fragment = any(len(norm(p.text)) >= 20 and fuzz.partial_ratio(norm(p.text), g) >= THRESHOLD for p in blocks)
            add(t, e.text, content, float(tight),
                "" if tight else "merged with other text" if loose
                else "only a fragment of the paragraph (split or truncated)" if fragment
                else "text present, not a paragraph" if content else "missing")
        elif t == "list":
            want = list(_items(e))
            got = [it for p in flat if p.type == "list" for it in _items(p)]
            content = sum(found(x, hay) for x, _, _ in want) / len(want)
            ok = [any(same(x, g) and d == gd for g, gd, _ in got) for x, d, _ in want]  # bullet or number: not judged
            miss = [x for (x, _, _), k in zip(want, ok) if not k]
            add(t, f"{len(want)} items", content, sum(ok) / len(want),
                "" if all(ok) else f"{len(miss)} item(s) not recovered as list items at the right depth, e.g. {miss[0]!r}")
        elif t == "table":
            texts = _gt_texts(e)
            content = sum(found(x, hay) for x in texts) / max(len(texts), 1)
            total = _count_cells(e)
            scored = [(i, *_best_hits(e, p)) for i, p in enumerate(tables)]
            fresh = [s for s in scored if s[0] not in used_tables] or scored
            if e.attrs.get("kind") == "multipage" and len(tables) >= 1:
                # a table over several pages is judged on its cells, however many pieces the tool wrote; whether
                # it came back as ONE table is a separate row (table_continuity), which a post-processing step can fix
                stitched, n_pieces = _stitch(e, tables)
                hits, tot, why = _best_hits(e, stitched)
                add(t, f"{e.attrs.get('rows')}x{e.attrs.get('cols')} table, {total} cells", content, hits / tot if tot else 0.0,
                    f"{hits:g}/{tot} cells at the right row and column over {n_pieces} piece(s)" if hits < tot else "")
                add("table_continuity", f"one table over the pages ({n_pieces} piece(s) found)", None, float(n_pieces == 1),
                    "" if n_pieces == 1 else f"written as {n_pieces} separate tables, one per page",
                    spec="The table below spans several pages and must come out as ONE table in the output.\n" + questions.spec(e))
            elif fresh:
                i, hits, tot, why = max(fresh, key=lambda s: s[1])
                used_tables.add(i)
                pieces = sum(1 for p in tables if sum(found(x, norm(" ".join(c.text for c in p.walk()))) for x in texts[:6]) >= 3)
                bits = [f"{hits:g}/{tot} cells at the right row and column"]
                if why["shift"]:
                    bits.append(f"rows shifted by {why['shift']:+d} (merged header rows?): half credit")
                if why["span"]:
                    bits.append(f"{why['span']} merged cell(s) lost")
                if why["nested"]:
                    bits.append(f"{why['nested']} nested table(s) not recovered as tables (flattened ones count half)")
                if pieces > 1:
                    bits.append(f"table split in {pieces} pieces")
                add(t, f"{e.attrs.get('rows')}x{e.attrs.get('cols')} table, {total} cells", content, hits / tot if tot else 0.0,
                    "; ".join(bits) if hits < tot else "")
            else:
                add(t, f"{e.attrs.get('rows')}x{e.attrs.get('cols')} table, {total} cells", content, 0.0,
                    "no table in the output" + (" (text present as plain text)" if content > 0.5 else ""))
        elif t == "link":
            content = float(found(e.text, hay))
            links = [p for p in flat if p.type == "link"]
            full = any((same(e.text, p.text) or norm(e.text) in norm(p.text)) and p.attrs["href"].rstrip("/") == e.attrs["href"].rstrip("/")
                       for p in links)  # the link may wrap more words than expected ("Details are in the reference document.")
            add(t, f"{e.text} -> {e.attrs['href']}", content, float(full),
                "" if full else "text found, target lost" if content else "missing")
        elif t == "footnote":
            content = float(found(e.text, hay))
            marker = re.escape(str(e.attrs["marker"]))
            lead = re.compile(rf"^[\[\(\^<]*(?:sup>)?{marker}(?:</sup>)?[\]\).:]?\s+(.+)$", re.S)
            explicit = any(p.type == "footnote" and same(e.text, p.text) for p in flat)
            # or its own block that starts with the note's number and then the note: how most extractors write it
            by_number = any(p.type in ("paragraph", "list_item") and any(
                (m := lead.match(line.strip())) and same(e.text, m.group(1)) for line in [p.text.strip(), *_note_lines(p.text)])
                for p in flat)
            add(t, f"[{e.attrs['marker']}] {e.text}", content, float(explicit or by_number),
                ("" if explicit else "" if by_number else "text present, not marked as a footnote" if content else "missing"))
        elif t in ("figure", "caption"):
            cap = e.text if t == "caption" else next((c.text for c in e.children if c.type == "caption"), "")
            content = float(found(cap, hay)) if cap else None
            idx = next((i for i, p in enumerate(predicted) if cap and (
                (p.text and covers(cap, p.text)) or
                any(c.type == "caption" and same(cap, c.text) for c in p.children) or
                (p.type == "list" and any(covers(cap, it.text) for it in p.children)))), None)  # "- Figure 1: ..."
            if idx is None:
                ok, why = False, "caption not found"
            else:
                near = [predicted[j] for j in (idx - 1, idx, idx + 1) if 0 <= j < len(predicted)]
                kinds = ("figure",) if t == "figure" else ("figure", "table")
                ok = any(p.type in kinds for p in near)
                why = "" if ok else "no " + " or ".join(kinds) + " next to the caption"
                words = e.attrs.get("image_text") if t == "figure" else None
                around = [predicted[j] for j in range(max(0, idx - 3), min(len(predicted), idx + 2)) if j != idx]
                if not ok and words and _chart_read(words, around):
                    ok, why = True, "chart returned as data (no figure marker)"
            add(t, cap or "figure", content, float(ok), why)
        elif t == "field":
            label, value = e.attrs["label"], e.attrs.get("value", "")
            spots = _context(hay, label)
            if not spots:
                add(t, f"{label}: {value}", 0.0, 0.0, "label missing")
            elif not value:
                add(t, f"{label}: (blank)", 1.0, 1.0, "blank field: only the label can be checked")
            else:
                others = [x for x in field_labels if x and x != norm(label) and norm(label) not in x and x not in norm(label)]
                cut = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in others) + r")(?!\w)") if others else None

                def after(m, cut=cut, value=value):  # the value must come before the next field's label
                    w = hay[m.end(): m.end() + len(norm(value)) + 40]
                    return w[:c.start()] if cut and (c := cut.search(w)) else w

                ok = any(found(value, after(m)) for m in spots)
                add(t, f"{label}: {value}", float(found(value, hay)), float(ok), "" if ok else "value not next to its label")
        elif t == "checkbox":
            want = bool(e.attrs.get("checked"))
            spots = _context(hay, e.text)
            if not spots:
                add(t, f"{'☒' if want else '☐'} {e.text}", 0.0, 0.0, "label missing")
            else:
                states = [_box_around(hay, m) for m in spots]
                ok = want in states
                add(t, f"{'☒' if want else '☐'} {e.text}", 1.0, float(ok),
                    "" if ok else "no tick mark found" if all(s is None for s in states) else "opposite state")
        elif t in ("header", "footer"):
            gone = not found(e.text, hay)
            add(t, e.text, None, float(gone), "" if gone else f"still in the output ({hay.count(norm(e.text))}x)")

    current.clear()
    paras = [e for e in gt.elements if e.type == "paragraph"]
    if len(paras) >= 2:
        pos = []
        for p in paras:
            n = norm(p.text)[:60]
            m = fuzz.partial_ratio_alignment(n, hay) if n else None
            pos.append(m.dest_start if m and m.score >= THRESHOLD else None)
        pairs = list(zip(pos, pos[1:]))
        right = sum(1 for a, b in pairs if a is not None and b is not None and a < b)
        add("reading_order", f"{len(paras)} paragraphs in order", None, right / len(pairs),
            "" if right == len(pairs) else f"{len(pairs) - right} of {len(pairs)} consecutive pairs out of order or missing",
            spec="These paragraphs must come in this order in the output (first words shown):\n"
                 + "\n".join(f"{i}. {p.text[:70]!r}" for i, p in enumerate(paras, 1)))
    _numbers_row(gt, hay, add)
    _title_row(gt, flat, hay, add, shift)
    _section_rows(gt, predicted, add)
    _page_row(gt, pages, add)
    return report
