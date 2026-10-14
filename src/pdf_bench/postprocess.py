"""Post-processing of a parser's output before scoring, the same for every parser.

Some structure is present in the output but not written the way the scorer reads it: a table cut at a page
break, a list printed as plain lines "1 Declare...", a section number that tells the heading level. Each step
rewrites the output text (Markdown or HTML) and nothing else, so raw and post-processed scores can be compared:
the difference is what a few generic rules recover, not what the parser did.

Steps never read the ground truth.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_ROW = re.compile(r"^\s*\|")
_BULLET = re.compile(r"^\s*(?:[•·▪◦‣–—]|\d+[\.\)]?)\s+(?=\S)")
_NUMBER = re.compile(r"^\s*(\d+)[\.\)]?\s+(.*)$")
_BULLET_ONLY = re.compile(r"^\s*[•·▪◦‣–—]\s+(.*)$")
_DASH_NUMBER = re.compile(r"^\s*-\s+(\d+)[\.\)]?\s+(.*)$")


# --- blocks -----------------------------------------------------------------------------------------------------

def _blocks(pages: list[str]) -> list[tuple[str, bool]]:
    """(text, starts_a_page) for every blank-line separated block of the output."""
    out: list[tuple[str, bool]] = []
    for page in pages:
        first = True
        for b in re.split(r"\n\s*\n", page.strip("\n")):
            if b.strip():
                out.append((b.strip("\n"), first))
                first = False
    return out


def _is_md_table(b: str) -> bool:
    lines = [x for x in b.split("\n") if x.strip()]
    return len(lines) >= 2 and all(_ROW.match(x) for x in lines)


def _is_html_table(b: str) -> bool:
    s = b.strip().lower()
    return s.startswith("<table") and s.endswith("</table>")


def _cells(row: str) -> list[str]:
    return [re.sub(r"[*_`]", "", c).strip().lower() for c in row.strip().strip("|").split("|")]


def _html_rows(b: str) -> list[str]:
    return re.findall(r"<tr\b.*?</tr>", b, flags=re.S | re.I)


def _html_text(row: str) -> list[str]:
    return [" ".join(re.sub(r"<[^>]+>", " ", c).split()).lower() for c in re.findall(r"<t[hd]\b.*?</t[hd]>", row, flags=re.S | re.I)]


def _md_width(b: str) -> int:
    return len(_cells(b.split("\n")[0]))


def _html_width(b: str) -> int:
    row = (_html_rows(b) or [""])[0]
    return sum(int((re.search(r'colspan="?(\d+)', c) or [0, 1])[1]) for c in re.findall(r"<t[hd]\b[^>]*>", row, flags=re.I))


# --- steps -------------------------------------------------------------------------------------------------------

def running_text(pages: list[str]) -> list[str]:
    """Drop short blocks that open or close several pages with the same text (running headers, footers, page numbers).

    Digits are ignored when comparing ("page 2" and "page 3" are the same line). Needs the output to be split by
    page: with a single text there is no repetition to see.
    """
    def key(b: str) -> str:
        return re.sub(r"\d+", "#", re.sub(r"[*_`#]", "", b)).strip().lower()

    split = [[b for b in re.split(r"\n\s*\n", p.strip("\n")) if b.strip()] for p in pages]
    if len([p for p in split if p]) < 2:
        return list(pages)
    seen: dict[str, int] = {}
    for blocks in split:
        for b in {key(x) for x in blocks[:2] + blocks[-2:] if len(x) <= 120 and "\n" not in x.strip()}:
            seen[b] = seen.get(b, 0) + 1
    gone = {k for k, n in seen.items() if n >= 2 and k}
    out = []
    for blocks in split:
        last = len(blocks) - 1
        out.append("\n\n".join(b for i, b in enumerate(blocks)
                               if not (i in (0, 1, last - 1, last) and len(b) <= 120 and "\n" not in b.strip() and key(b) in gone)))
    return out


def merge_tables(pages: list[str]) -> list[str]:
    """Join two tables that are one table cut by a page break (or whose header is repeated).

    Two consecutive tables are merged when the second starts a page and has the same number of columns, or when
    the second starts with the first one's header row. A repeated header row is dropped. The result is a single
    text (the page boundaries are no longer needed afterwards).
    """
    out: list[tuple[str, bool]] = []
    for text, starts in _blocks(pages):
        if out and _is_md_table(out[-1][0]) and _is_md_table(text):
            prev = out[-1][0]
            same_head = _cells(prev.split("\n")[0]) == _cells(text.split("\n")[0])
            if same_head or (starts and _md_width(prev) == _md_width(text)):
                rows = [x for x in text.split("\n") if x.strip()]
                body = rows[2:] if same_head else [r for i, r in enumerate(rows) if i != 1]  # drop the separator (and the repeated header)
                out[-1] = (prev + "\n" + "\n".join(body), out[-1][1])
                continue
        if out and _is_html_table(out[-1][0]) and _is_html_table(text):
            prev = out[-1][0]
            pr, nr = _html_rows(prev), _html_rows(text)
            if pr and nr:
                same_head = _html_text(pr[0]) == _html_text(nr[0])
                if same_head or (starts and _html_width(prev) == _html_width(text)):
                    keep = nr[1:] if same_head else nr
                    out[-1] = (re.sub(r"</tbody>\s*</table>\s*$|</table>\s*$", lambda m: "\n".join(keep) + "\n" + m.group(0), prev, count=1,
                                      flags=re.I), out[-1][1])
                    continue
        out.append((text, starts))
    return ["\n\n".join(t for t, _ in out)]


def lists(pages: list[str]) -> list[str]:
    """Plain lines that start with a bullet glyph or a number become a list.

    A run of two or more such lines counts as a list, whether they are one block or one block per line (some tools
    separate every item with a blank line); a numbered run must start at 1. A number that restarts at
    1 opens a nested list, a number that continues an outer sequence closes the inner one. A flat list whose items
    are "- 1 text" (numbers kept as text, indentation lost) is nested the same way.
    """
    def glyph_runs(lines: list[str]) -> str | None:
        """Lines starting with a bullet glyph inside a block of running text ("Intro.\n• a\n• b"): the run is a list,
        the other lines stay a paragraph. None when the block has no such line, or all of it is one."""
        flags = [bool(_BULLET_ONLY.match(x)) for x in lines]
        if not any(flags) or all(flags):
            return None
        parts, cur, mode = [], [], None
        for line, is_bullet in zip(lines, flags):
            if is_bullet != mode and cur:
                parts.append((mode, cur))
                cur = []
            mode = is_bullet
            cur.append(line)
        parts.append((mode, cur))
        return "\n\n".join("\n".join("- " + _BULLET_ONLY.match(x).group(1) for x in cur) if bullets else "\n".join(cur)
                            for bullets, cur in parts)

    def convert(block: str) -> str:
        lines = block.split("\n")
        if len(lines) == 1 and _BULLET_ONLY.match(lines[0]):
            return "- " + _BULLET_ONLY.match(lines[0]).group(1)  # a lone bullet line is a list of one item
        if len(lines) >= 2 and (mixed := glyph_runs(lines)) is not None and not any(x.lstrip().startswith(("|", "<")) for x in lines):
            return mixed
        if len(lines) >= 2 and all(_DASH_NUMBER.match(x) for x in lines):
            lines = [f"{m.group(1)} {m.group(2)}" for m in map(_DASH_NUMBER.match, lines)]
        elif any(x.lstrip().startswith(("-", "*", "+", "|", "#", "<", ">", "[^")) for x in lines) or len(lines) < 2:
            return block
        if all(_BULLET_ONLY.match(x) for x in lines):
            return "\n".join("- " + _BULLET_ONLY.match(x).group(1) for x in lines)
        nums = [_NUMBER.match(x) for x in lines]
        if not all(nums) or int(nums[0].group(1)) != 1:
            return block
        stack: list[list[int]] = []  # [last number, items so far] at each depth
        out = []
        for m in nums:
            n = int(m.group(1))
            if not stack:
                stack = [[n, 1]]
            else:
                outer = next((d for d in range(len(stack) - 1) if n == stack[d][0] + 1), None)
                inner = n == stack[-1][0] + 1
                if inner and (outer is None or stack[-1][1] < 2):
                    stack[-1] = [n, stack[-1][1] + 1]
                elif outer is not None:  # continues an outer list: the inner one is over
                    del stack[outer + 1:]
                    stack[outer] = [n, stack[outer][1] + 1]
                elif n == 1:
                    stack.append([1, 1])
                else:
                    return block  # numbers that do not follow each other: not a list
            out.append("   " * (len(stack) - 1) + "- " + m.group(2))
        return "\n".join(out)

    def page(p: str) -> str:
        out: list[str] = []
        run: list[str] = []

        def flush(tail: bool = False):
            # numbered lines closing a page are footnotes more often than a list: a short run there stays as it was
            footnotes = tail and len(run) < 4 and not any(_BULLET_ONLY.match(x) for x in run)
            if len(run) >= 2 and not footnotes:  # one line per block: join them so the run is read as a list, or put them back
                joined = "\n".join(run)
                done = convert(joined)
                out.extend([done] if done != joined else run)
            else:
                out.extend(convert(b) for b in run)
            run.clear()

        for b, _ in _blocks([p]):
            if "\n" not in b.strip() and (_BULLET.match(b) or _DASH_NUMBER.match(b)):
                run.append(b)
                continue
            flush()
            out.append(convert(b))
        flush(tail=True)
        return "\n\n".join(out)

    return [page(p) for p in pages]


_SECTION = re.compile(r"^(\d+(?:\.\d+)*)[\.\)]?\s+\S")


def _level(text: str) -> int | None:
    m = _SECTION.match(text.strip())
    return None if not m else min(m.group(1).count(".") + 2, 6)


def heading_levels(pages: list[str]) -> list[str]:
    """Heading level from the section number in its text: "2. Method" is H2, "1.1 Motor" H3, "1.1.1" H4.

    Headings without a number keep the level the parser gave them.
    """
    def md(m: re.Match) -> str:
        title = m.group(2).replace("**", "")
        lv = _level(title)
        return f"{'#' * lv} {title}" if lv else m.group(0)

    def html(m: re.Match) -> str:
        lv = _level(re.sub(r"<[^>]+>", "", m.group(3)))
        return f"<h{lv}{m.group(2)}>{m.group(3)}</h{lv}>" if lv else m.group(0)

    out = []
    for p in pages:
        p = re.sub(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", md, p, flags=re.M)
        p = re.sub(r"<h([1-6])([^>]*)>(.*?)</h\1>", html, p, flags=re.S | re.I)
        out.append(p)
    return out


STEPS: dict[str, tuple[str, Callable[[list[str]], list[str]]]] = {
    "running_text": ("Drop running headers, footers and page numbers repeated on several pages", running_text),
    "merge_tables": ("Merge a table cut by a page break", merge_tables),
    "lists": ("Turn plain numbered or bulleted lines into lists", lists),
    "heading_levels": ("Heading level from the section number (2. is H2, 1.1 is H3)", heading_levels),
}


# kinds a step can create from text, so a tool that "cannot express" them is still scored on them after that step
CREATES = {"lists": {"list"}}


def apply(pages: list[str], steps: list[str] | tuple[str, ...]) -> list[str]:
    """The output after the chosen steps, in the order of STEPS. Unknown names are ignored."""
    out = list(pages)
    for name, (_, fn) in STEPS.items():
        if name in steps:
            out = fn(out)
    return out
