"""Synthetic PDFs for the structured-extraction benchmark, with a ground truth in the element model.

The PDF and its ground truth come from the same calls (:class:`Builder`), so they cannot drift apart.
Two families:

* **one document per category** (headings, lists, simple / borderless / merged / nested / multi-page
  tables, links, footnotes, figures, forms, columns, running header and footer, a scan): the ground
  truth lists only the element type under test, the rest of the page is filler that is not asserted;
* **composite documents** (an accident report form, an annual report) mixing many types.

Layout choices are deliberate: headings are told apart by size and weight only (no PDF outline),
form fields are plain text with a rule under the value, checkboxes are drawn shapes, scans have no
text layer. A parser has to infer the structure, as it would on real documents.
Generated PDFs are proxies: they are cleaner than real ones.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, FrameBreak, Image as RLImage, KeepTogether,
                                ListFlowable, ListItem, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

from .elements import Element, GroundTruth

HEADER = "ACME Insurance - File 2025"
FOOTER = "Confidential document - page"

_BASE = getSampleStyleSheet()["BodyText"]
BODY = ParagraphStyle("body", parent=_BASE, fontName="Helvetica", fontSize=10, leading=14, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
HEAD = {1: ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=20, leading=24, spaceBefore=10, spaceAfter=8),
        2: ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=15, leading=19, spaceBefore=8, spaceAfter=6),
        3: ParagraphStyle("h3", parent=BODY, fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=6, spaceAfter=4)}
CAPTION = ParagraphStyle("caption", parent=BODY, fontName="Helvetica-Oblique", fontSize=9, leading=11)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=8, leading=10, spaceAfter=2)

WORDS = ("file contract claim coverage insured vehicle declaration appraisal compensation deductible "
         "premium renewal amendment cancellation certificate voucher settlement deadline refund "
         "driver third-party liability damage report").split()


def lorem(rng: random.Random, n: int = 40) -> str:
    """Neutral insurance-flavoured English filler of about ``n`` words, one sentence per ~12 words."""
    words = [rng.choice(WORDS) for _ in range(n)]
    sentences = [" ".join(words[i:i + 12]).capitalize() + "." for i in range(0, n, 12)]
    return " ".join(sentences)


# --- table specs ----------------------------------------------------------------------------------

@dataclass
class Check:
    """A drawn checkbox (ticked or not) placed in a table cell."""

    checked: bool


@dataclass
class Cell:
    """A cell that holds text and/or a nested table."""

    text: str = ""
    table: "TableSpec | None" = None


@dataclass
class TableSpec:
    rows: list[list]  # str | Cell | Check | TableSpec (a bare TableSpec is a cell holding only a table)
    header_rows: int = 1
    spans: tuple = ()  # ((col0, row0, col1, row1), ...) inclusive, reportlab convention
    grid: bool = True
    widths: tuple | None = None
    repeat_header: bool = False


class CheckBox(Flowable):
    """A square, crossed when ``checked``; drawn, so it has no text for a parser to read."""

    def __init__(self, checked: bool, size: float = 11):
        super().__init__()
        self.checked, self.size = checked, size
        self.width = self.height = size

    def wrap(self, *_):
        return self.size, self.size

    def draw(self):
        c = self.canv
        c.setLineWidth(0.8)
        c.rect(0, 0, self.size, self.size)
        if self.checked:
            c.setLineWidth(1.4)
            c.line(2, 2, self.size - 2, self.size - 2)
            c.line(2, self.size - 2, self.size - 2, 2)


class NoteAnchor(Flowable):
    """Zero-size flowable that records on which page a footnote is cited."""

    def __init__(self, builder: "Builder", marker: str, text: str):
        super().__init__()
        self.builder, self.marker, self.text = builder, marker, text

    def wrap(self, *_):
        return 0, 0

    def draw(self):
        self.builder.page_notes.setdefault(self.canv.getPageNumber(), []).append((self.marker, self.text))


# --- builder ----------------------------------------------------------------------------------------

class Builder:
    """Collects flowables for the PDF and, in step, the elements of the ground truth."""

    def __init__(self, name: str, category: str, tested: set[str], *, seed: int = 0, columns: int = 1,
                 running: bool = False):
        self.name, self.category, self.tested = name, category, set(tested)
        self.seed = seed
        self.rng = random.Random(f"{name}-{seed}")
        self.columns, self.running = columns, running
        self.story: list = []
        self.elements: list[Element] = []
        self.page_notes: dict[int, list[tuple[str, str]]] = {}
        self.figures = 0
        self.meta: dict = {}

    # --- variation: seed 0 is the reference document, other seeds change the content, not the layout
    def vary(self, base: int, spread: int) -> int:
        return base if self.seed == 0 else max(2, base + self.rng.randint(-spread, spread))

    def values(self, base: list[int]) -> list[int]:
        return list(base) if self.seed == 0 else [max(5, int(v * self.rng.uniform(0.6, 1.4))) for v in base]

    def person(self) -> tuple[str, str]:
        return PEOPLE[self.seed % len(PEOPLE)]

    def date(self, base: str = "12/03/2025") -> str:
        if self.seed == 0:
            return base
        return f"{self.rng.randint(1, 28):02d}/{self.rng.randint(1, 12):02d}/2025"

    def policy(self, base: str = "C-4410-77") -> str:
        return base if self.seed == 0 else f"C-{self.rng.randint(1000, 9999)}-{self.rng.randint(10, 99)}"

    def flags(self, base: list[bool]) -> list[bool]:
        """Checkbox states; other seeds draw new ones but always keep one ticked and one empty box."""
        if self.seed == 0:
            return list(base)
        out = [self.rng.random() < 0.5 for _ in base]
        if len(out) > 1 and all(out):
            out[self.rng.randrange(len(out))] = False
        if len(out) > 1 and not any(out):
            out[self.rng.randrange(len(out))] = True
        return out

    # ground truth is limited to the types under test
    def gt(self, el: Element) -> None:
        if el.type in self.tested:
            self.elements.append(el)

    def filler(self, words: int = 40, asserted: bool = False) -> None:
        """Body text; with ``asserted`` it is also in the ground truth (needed for section_path and page)."""
        text = lorem(self.rng, words)
        self.story.append(Paragraph(escape(text), BODY))
        if asserted:
            self.gt(Element("paragraph", text))

    def heading(self, text: str, level: int = 1) -> None:
        self.story.append(Paragraph(escape(text), HEAD[level]))
        self.gt(Element("heading", text, level))

    def paragraph(self, text: str) -> None:
        self.story.append(Paragraph(escape(text), BODY))
        self.gt(Element("paragraph", text))

    def bullets(self, items: list, ordered: bool = False) -> None:
        flow, el = self._list(items, ordered)
        self.story += [flow, Spacer(1, 6)]
        self.gt(el)

    def _list(self, items: list, ordered: bool):
        flows, children = [], []
        for item in items:
            text, sub = (item, None) if isinstance(item, str) else item
            content = [Paragraph(escape(text), BODY)]
            kids = []
            if sub:
                f, e = self._list(sub, ordered)
                content.append(f)
                kids.append(e)
            flows.append(ListItem(content, leftIndent=18))
            children.append(Element("list_item", text, children=kids))
        flow = ListFlowable(flows, bulletType="1" if ordered else "bullet",
                            start=None if ordered else "•", leftIndent=18, bulletFontSize=9)
        return flow, Element("list", attrs={"ordered": ordered}, children=children)

    def table(self, spec: TableSpec, kind: str = "simple", caption: str | None = None) -> None:
        flow, el = self._table(spec)
        el.attrs["kind"] = kind
        self.story += [flow, Spacer(1, 8)]
        if caption:
            self.story.append(Paragraph(escape(caption), CAPTION))
        self.gt(el)
        if caption:
            self.gt(Element("caption", caption))

    def _table(self, spec: TableSpec, inner: bool = False):
        covered = set()
        anchors = {}
        for c0, r0, c1, r1 in spec.spans:
            anchors[(c0, r0)] = (r1 - r0 + 1, c1 - c0 + 1)
            covered |= {(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)} - {(c0, r0)}
        data, cells = [], []
        for r, row in enumerate(spec.rows):
            out = []
            for c, value in enumerate(row):
                if (c, r) in covered:
                    out.append("")
                    continue
                head = r < spec.header_rows
                style = CELLB if head else CELL
                children, text = [], ""
                if isinstance(value, TableSpec):
                    value = Cell(table=value)
                if isinstance(value, Check):
                    out.append(CheckBox(value.checked))
                    children.append(Element("checkbox", attrs={"checked": value.checked}))
                elif isinstance(value, Cell):
                    flows = []
                    if value.text:
                        flows.append(Paragraph(escape(value.text), style))
                        text = value.text
                    if value.table:
                        flow, sub = self._table(value.table, inner=True)
                        flows.append(flow)
                        children.append(sub)
                    out.append(flows)
                else:
                    text = str(value)
                    out.append(Paragraph(escape(text), style) if text else "")
                rs, cs = anchors.get((c, r), (1, 1))
                attrs = {"r": r, "c": c, "header": head}
                if rs > 1:
                    attrs["rowspan"] = rs
                if cs > 1:
                    attrs["colspan"] = cs
                cells.append(Element("cell", text, attrs=attrs, children=children))
            data.append(out)
        n_cols = max(len(r) for r in spec.rows)
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                 ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if spec.grid:
            style.append(("GRID", (0, 0), (-1, -1), 0.6, colors.black))
        if spec.header_rows and spec.grid:
            style.append(("BACKGROUND", (0, 0), (-1, spec.header_rows - 1), colors.HexColor("#e6e6e6")))
        elif spec.header_rows:
            style.append(("LINEBELOW", (0, spec.header_rows - 1), (-1, spec.header_rows - 1), 0.8, colors.black))
        style += [("SPAN", (c0, r0), (c1, r1)) for c0, r0, c1, r1 in spec.spans]
        widths = spec.widths or (None if inner else [(A4[0] - 4 * cm) / n_cols] * n_cols)
        flow = Table(data, colWidths=widths, repeatRows=spec.header_rows if spec.repeat_header else 0)
        flow.setStyle(TableStyle(style))
        return flow, Element("table", attrs={"rows": len(spec.rows), "cols": n_cols}, children=cells)

    def link(self, before: str, label: str, url: str, after: str = "") -> None:
        self.story.append(Paragraph(
            f'{escape(before)} <a href="{escape(url)}" color="blue"><u>{escape(label)}</u></a>{escape(after)}', BODY))
        self.gt(Element("link", label, attrs={"href": url}))

    def footnote(self, before: str, marker: str, note: str, after: str = "") -> None:
        self.story.append(Paragraph(f"{escape(before)}<super>{escape(marker)}</super>{escape(after)}", BODY))
        self.story.append(NoteAnchor(self, marker, note))
        self.gt(Element("footnote", note, attrs={"marker": marker}))

    def figure(self, caption: str, title: str, values: list[int], labels: list[str]) -> None:
        self.figures += 1
        png = chart_png(title, values, labels)
        img = RLImage(io.BytesIO(png), width=11 * cm, height=6.5 * cm)
        self.story.append(KeepTogether([img, Paragraph(escape(caption), CAPTION), Spacer(1, 6)]))
        self.gt(Element("figure", attrs={"image_text": [title, *labels]},
                        children=[Element("caption", caption)]))

    def fields(self, items: list[tuple[str, str]], name_width: float = 5 * cm) -> None:
        """Label, then the value written on a rule (a blank value leaves just the rule)."""
        rows = [[Paragraph(f"<b>{escape(label)}</b>", CELL), Paragraph(escape(value), CELL) if value else ""]
                for label, value in items]
        t = Table(rows, colWidths=[name_width, A4[0] - 4 * cm - name_width], rowHeights=[0.8 * cm] * len(rows))
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                               ("LINEBELOW", (1, 0), (1, -1), 0.5, colors.black)]))
        self.story += [t, Spacer(1, 8)]
        for label, value in items:
            self.gt(Element("field", attrs={"label": label, "value": value}))

    def checkboxes(self, items: list[tuple[str, bool]]) -> None:
        rows = [[CheckBox(on), Paragraph(escape(label), CELL)] for label, on in items]
        t = Table(rows, colWidths=[0.8 * cm, 9 * cm], hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        self.story += [t, Spacer(1, 8)]
        for label, on in items:
            self.gt(Element("checkbox", label, attrs={"checked": on}))

    def page_break(self) -> None:
        self.story.append(PageBreak())

    def column_break(self) -> None:
        self.story.append(FrameBreak())

    # --- output
    def build(self, pdf: Path) -> GroundTruth:
        if self.running:
            self.gt(Element("header", HEADER))
            self.gt(Element("footer", FOOTER))

        def page(canv, doc):
            canv.saveState()
            if self.running:
                canv.setFont("Helvetica", 9)
                canv.drawString(2 * cm, A4[1] - 1.2 * cm, HEADER)
                canv.drawString(2 * cm, 1.0 * cm, f"{FOOTER} {canv.getPageNumber()}")
            canv.restoreState()

        def page_end(canv, doc):
            notes = self.page_notes.get(canv.getPageNumber(), [])
            if notes:
                canv.saveState()
                canv.setLineWidth(0.5)
                y = 2.6 * cm + 10 * len(notes)
                canv.line(2 * cm, y + 6, 6 * cm, y + 6)
                canv.setFont("Helvetica", 8)
                for marker, text in notes:
                    canv.drawString(2 * cm, y - 4, f"{marker} {text}")
                    y -= 10
                canv.restoreState()

        w, h = A4
        top, bottom = 2.0 * cm, 3.2 * cm
        gap = 0.8 * cm
        col_w = (w - 4 * cm - gap * (self.columns - 1)) / self.columns
        frames = [Frame(2 * cm + i * (col_w + gap), bottom, col_w, h - top - bottom, id=f"f{i}")
                  for i in range(self.columns)]
        doc = BaseDocTemplate(str(pdf), pagesize=A4, title=self.name, author="pdf-bench",
                              pageTemplates=[PageTemplate(id="p", frames=frames, onPage=page, onPageEnd=page_end)])
        doc.build(self.story)
        return GroundTruth(self.name, self.category, sorted(self.tested), self.elements, self.meta)


# --- images -----------------------------------------------------------------------------------------

def chart_png(title: str, values: list[int], labels: list[str]) -> bytes:
    """A small bar chart, raster: its labels are pixels, not text a parser can extract without OCR."""
    img = Image.new("RGB", (660, 390), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 12), title, fill="black")
    top = max(values)
    for i, (v, lab) in enumerate(zip(values, labels)):
        x = 60 + i * 130
        bar = int(260 * v / top)
        d.rectangle([x, 330 - bar, x + 80, 330], fill=(70 + 40 * i % 180, 110, 170), outline="black")
        d.text((x + 28, 334 - bar - 14), str(v), fill="black")
        d.text((x + 10, 340), lab, fill="black")
    d.line([40, 330, 640, 330], fill="black")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def to_scan(pdf: Path, dpi: int = 150, seed: int = 0) -> None:
    """Replace ``pdf`` by an image-only version: no text layer, light skew and noise like a scanner."""
    rng = random.Random(seed)
    src = pymupdf.open(pdf)
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = img.rotate(rng.uniform(-0.8, 0.8), resample=Image.BICUBIC, fillcolor="white")
        noise = Image.effect_noise(img.size, 18).convert("RGB")
        img = Image.blend(img.filter(ImageFilter.GaussianBlur(0.6)), noise, 0.06)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    src.close()
    out.save(pdf)
    out.close()


# --- documents ---------------------------------------------------------------------------------------

PEOPLE = [("Martin", "Claire"), ("Bernard", "Lucas"), ("Dubois", "Emma"), ("Moreau", "Hugo"),
          ("Laurent", "Léa"), ("Simon", "Nathan"), ("Michel", "Jade"), ("Garcia", "Louis")]


def amount(rng: random.Random) -> str:
    """A money amount in one of the styles that parsers tend to garble: "12 480,50", "9,800.25", "845,5", "1 020"."""
    v, cents = rng.randint(120, 98000), rng.randint(0, 99)
    style = rng.randrange(4)
    if style == 0:
        return f"{v:,}".replace(",", " ") + f",{cents:02d}"
    if style == 1:
        return f"{v:,}.{cents:02d}"
    if style == 2:
        return f"{v},{cents // 10}"
    return f"{v:,}".replace(",", " ")


def _sinistres(rng: random.Random, n: int, offset: int = 0) -> list[list[str]]:
    rows = []
    for i in range(n):
        nom, prenom = PEOPLE[(i + offset) % len(PEOPLE)]
        rows.append([f"S-{2025000 + i * 7}", f"{nom} {prenom}", amount(rng) + " EUR",
                     rng.choice(["Open", "Closed", "Under appraisal"])])
    return rows


def doc_headings(seed: int = 0) -> Builder:
    b = Builder("headings", "heading", {"heading", "paragraph"}, seed=seed)
    b.heading("Claims management report", 1)
    b.filler(asserted=True)
    b.heading("1. Scope of the study", 2)
    b.filler(asserted=True)
    b.heading("1.1 Motor policies", 3)
    b.filler(30, asserted=True)
    b.heading("1.2 Home policies", 3)
    b.filler(30, asserted=True)
    b.page_break()
    b.heading("2. Method", 2)
    b.filler(asserted=True)
    b.heading("3. Results", 2)
    b.filler(asserted=True)
    b.heading("3.1 Processing times", 3)
    b.filler(30, asserted=True)
    b.heading("Conclusion", 1)
    b.filler(asserted=True)
    return b


def doc_lists(seed: int = 0) -> Builder:
    b = Builder("lists", "list", {"list"}, seed=seed)
    b.heading("Documents to provide", 2)
    b.bullets(["Declarant's identity document", "Bank account details",
               ("Claim evidence", ["Photographs of the damage", "Repair estimate",
                                              "Purchase invoice of the item"]),
               "Certificate of current insurance"])
    b.filler()
    b.heading("Steps of the procedure", 2)
    b.bullets(["Declare the claim within five days",
               ("Have the damage assessed", ["Contact the appointed expert", "Keep the damaged items"]),
               "Send the supporting documents", "Wait for the compensation decision"], ordered=True)
    b.filler()
    return b


def _sinistre_spec(b: "Builder", n: int = 6, **kw) -> TableSpec:
    return TableSpec([["Reference", "Insured", "Amount", "Status"], *_sinistres(b.rng, b.vary(n, 2), b.seed)], **kw)


def doc_table_simple(seed: int = 0) -> Builder:
    b = Builder("table_simple", "table_simple", {"table"}, seed=seed)
    b.heading("Claims of the quarter", 2)
    b.filler(30)
    b.table(_sinistre_spec(b), "simple", caption=None)
    b.filler()
    return b


def doc_table_borderless(seed: int = 0) -> Builder:
    b = Builder("table_borderless", "table_borderless", {"table"}, seed=seed)
    b.heading("Deductible schedule", 2)
    b.filler(30)
    b.table(TableSpec([["Coverage", "Deductible", "Limit"], ["Glass breakage", "50 EUR", "800 EUR"],
                       ["Theft", "250 EUR", "15,000 EUR"], ["Fire", "150 EUR", "30,000 EUR"],
                       ["Water damage", "200 EUR", "10,000 EUR"]], grid=False, widths=(6 * cm, 4 * cm, 4 * cm)),
            "borderless")
    b.filler()
    return b


def doc_table_merged(seed: int = 0) -> Builder:
    b = Builder("table_merged", "table_merged", {"table"}, seed=seed)
    b.heading("Premiums by area and plan", 2)
    rows = [["Area", "Third-party plan", "", "Comprehensive plan", ""],
            ["", "Monthly", "Annual", "Monthly", "Annual"],
            ["North", "32 EUR", "360 EUR", "58 EUR", "640 EUR"],
            ["South", "29 EUR", "330 EUR", "52 EUR", "590 EUR"],
            ["Greater Paris", "41 EUR", "460 EUR", "77 EUR", "850 EUR"],
            ["Total (average)", "34 EUR", "383 EUR", "62 EUR", "693 EUR"]]
    spans = ((0, 0, 0, 1), (1, 0, 2, 0), (3, 0, 4, 0))
    b.table(TableSpec(rows, header_rows=2, spans=spans), "merged")
    b.filler()
    return b


def doc_table_nested(seed: int = 0) -> Builder:
    b = Builder("table_nested", "table_nested", {"table"}, seed=seed)
    b.heading("Coverage details by policy", 2)
    garanties = lambda *xs: TableSpec([["Coverage", "Limit"], *[[g, p] for g, p in xs]])  # noqa: E731
    rows = [["Policy", "Holder", "Included coverage"],
            ["C-4410", "Martin Claire", garanties(("Liability", "unlimited"), ("Glass breakage", "800 EUR"))],
            ["C-4411", "Bernard Lucas", garanties(("Liability", "unlimited"), ("Theft", "15,000 EUR"),
                                                  ("Fire", "30,000 EUR"))]]
    b.table(TableSpec(rows, widths=(3 * cm, 4 * cm, 10 * cm)), "nested")
    b.filler()
    return b


def doc_table_multipage(seed: int = 0) -> Builder:
    b = Builder("table_multipage", "table_multipage", {"table"}, seed=seed)
    b.heading("Claims register of the year", 2)
    b.table(TableSpec([["Reference", "Insured", "Amount", "Status"], *_sinistres(b.rng, 48 if b.seed == 0 else b.rng.randint(48, 70), b.seed)], repeat_header=True),
            "multipage")
    return b


def doc_numbers(seed: int = 0) -> Builder:
    """Figures that look alike: thousands separators, decimals, percentages, negatives, times, near-identical values."""
    b = Builder("numbers", "numbers", {"heading", "paragraph", "table"}, seed=seed)
    b.heading("Settlement statement", 1)
    b.paragraph(f"The insurer paid {amount(b.rng)} EUR on {b.date('03/11/2025')} at 14:35, "
                f"after a deductible of {amount(b.rng)} EUR; the file number is {b.rng.randint(100, 999)}-{b.rng.randint(1000, 9999)}.")
    b.heading("Indicators", 2)
    base = ["12 480,50", "12 408,50", "124 805,00", "9 800,25", "1 020", "10 200"]
    if b.seed:
        base = [amount(b.rng) for _ in base]
    rows = [["Indicator", "2024", "2025", "Change"],
            ["Premiums written", base[0], base[1], "+3,2 %"],
            ["Claims paid", base[2], base[3], "-12,5 %"],
            ["Reserves", base[4], base[5], "0,0 %"],
            ["Settlement time", "14:35", "09:05", "-5h30"]]
    b.table(TableSpec(rows), "simple")
    b.paragraph(f"Call 04 72 {b.rng.randint(10, 99)} {b.rng.randint(10, 99)} {b.rng.randint(10, 99)} before 18:00 "
                f"or write before the 15/01/2026.")
    return b


def doc_links(seed: int = 0) -> Builder:
    b = Builder("links", "link", {"link"}, seed=seed)
    b.heading("Useful resources", 2)
    b.link("The claim form is available on", "the customer portal", "https://assures.acme.example/declaration", ".")
    b.filler()
    b.link("To contest a decision, see", "the complaints procedure", "https://acme.example/reclamation/procedure")
    b.filler()
    b.link("The mediator can be reached at", "mediation@acme.example", "mailto:mediation@acme.example", " as a last resort.")
    return b


def doc_footnotes(seed: int = 0) -> Builder:
    b = Builder("footnotes", "footnote", {"footnote"}, seed=seed)
    b.heading("Compensation", 2)
    b.footnote("The compensation is calculated after the deductible is subtracted", "1",
               "The deductible is set in the contract and depends on the coverage.", ".")
    b.filler()
    b.footnote("The payment period starts when the appraisal report is delivered", "2",
               "Report written by an accredited independent expert.", ".")
    b.filler()
    return b


def doc_figures(seed: int = 0) -> Builder:
    b = Builder("figures", "figure", {"figure"}, seed=seed)
    b.heading("Claim trends", 2)
    b.filler(30)
    b.figure("Figure 1: Claims reported per quarter in 2025", "Claims reported", b.values([120, 95, 140, 110]),
             ["T1", "T2", "T3", "T4"])
    b.filler(30)
    b.figure("Figure 2: Claims by coverage", "Breakdown", b.values([210, 90, 60]),
             ["Motor", "Home", "Health"])
    return b


def doc_forms(seed: int = 0) -> Builder:
    b = Builder("forms", "form", {"field", "checkbox"}, seed=seed)
    b.heading("Claim declaration", 2)
    last, first = b.person()
    b.fields([("Last name", last), ("First name", first), ("Policy no.", b.policy()), ("Date of loss", b.date()),
              ("Place", "Lyon, 3rd district"), ("Witness, if any", "")])
    flags = b.flags([False, True, True, False])
    b.checkboxes(list(zip(["Injuries were reported", "A joint report was drawn up",
                           "Police attended the scene", "The vehicle is immobilised"], flags)))
    b.fields([("Declarant signature", "")])
    return b


def doc_columns(seed: int = 0) -> Builder:
    b = Builder("columns", "reading_order", {"paragraph"}, columns=2, seed=seed)
    texts = [f"Section {i}. " + lorem(b.rng, 36) for i in range(1, 7)]
    for i, t in enumerate(texts):
        b.paragraph(t)
    return b


def doc_running(seed: int = 0) -> Builder:
    b = Builder("running", "header_footer", {"header", "footer"}, running=True, seed=seed)
    b.heading("General terms", 1)
    for _ in range(3):
        b.filler(60)
        b.heading("Article", 3)
        b.filler(60)
        b.page_break()
    return b


def doc_mini(seed: int = 0) -> Builder:
    """Heading, paragraph, list and table: the source of the scanned document."""
    b = Builder("scan", "scan", {"heading", "paragraph", "list", "table"}, seed=seed)
    b.heading("Renewal notice", 1)
    b.paragraph("Your policy expires on 31 December 2025 and will be renewed automatically.")
    b.bullets(["Annual premium: 640 EUR", "Deductible: 150 EUR", "Roadside assistance included"])
    b.table(_sinistre_spec(b, 4), "simple")
    b.meta["text_layer"] = False
    return b


def doc_accident_report(seed: int = 0) -> Builder:
    """Composite: an amicable accident report with nested tables, ticked boxes and a sketch."""
    b = Builder("constat", "composite", {"heading", "field", "checkbox", "table", "figure", "caption"}, seed=seed)
    b.heading("Joint motor accident report", 1)
    b.fields([("Date of accident", b.date()), ("Time", "5:45 pm"), ("Place", "Republic Street, Lyon"),
              ("Witness", "")])
    b.checkboxes(list(zip(["Injuries, even slight", "Property damage other than to vehicles A and B"],
                          b.flags([False, True]))))

    def veh(nom, prenom, immat, assureur, contrat):
        return TableSpec([["Last name", nom], ["First name", prenom], ["Plate", immat], ["Insurer", assureur],
                          ["Policy no.", contrat]], header_rows=0)

    b.heading("Vehicles involved", 2)
    b.table(TableSpec([["Vehicle A", "Vehicle B"],
                       [veh(*PEOPLE[b.seed % len(PEOPLE)], "AB-123-CD", "ACME", b.policy("C-4410")),
                        veh(*PEOPLE[(b.seed + 1) % len(PEOPLE)], "EF-456-GH", "North Mutual", b.policy("M-9921"))]],
                      widths=((A4[0] - 4 * cm) / 2,) * 2), "nested")
    b.heading("Circumstances", 2)
    rows = [["Circumstance", "A", "B"]]
    labels = ["Was parked", "Was leaving a parking space", "Was reversing", "Was changing lanes",
              "Had not seen a priority sign"]
    base = [(False, False), (False, True), (True, False), (False, False), (False, True)]
    marks = b.flags([x for pair in base for x in pair])
    for k, label in enumerate(labels):
        rows.append([label, Check(marks[2 * k]), Check(marks[2 * k + 1])])
    b.table(TableSpec(rows, widths=(11 * cm, 3 * cm, 3 * cm)), "form")
    b.figure("Sketch of the accident", "Sketch", b.values([3, 2]), ["A", "B"])
    b.heading("Signatures", 2)
    b.fields([("Driver A signature", ""), ("Driver B signature", "")])
    return b


def doc_annual_report(seed: int = 0) -> Builder:
    """Composite: a report mixing headings, lists, links, a note, a merged-header table and a figure."""
    b = Builder("rapport", "composite",
                {"heading", "paragraph", "list", "link", "footnote", "table", "figure", "caption", "header", "footer"},
                running=True, seed=seed)
    b.heading("Annual report 2025", 1)
    b.filler(50, asserted=True)
    b.heading("1. Highlights", 2)
    b.bullets(["Portfolio up 8%", ("Three new branches opened", ["Lille", "Nantes", "Dijon"]),
               "Claim declaration journey redesigned"])
    b.footnote("The portfolio excludes group contracts", "1", "Source: technical department, January 2026.", ".")
    b.link("Details are in", "the reference document", "https://acme.example/reference-2025", ".")
    b.heading("2. Results", 2)
    rows = [["Indicator", "2024", "", "2025", ""],
            ["", "Value", "Rate", "Value", "Rate"],
            ["Claims reported", "1,840", "6.1%", "1,920", "6.0%"],
            ["Average delay (days)", "21", "-", "17", "-"]]
    b.table(TableSpec(rows, header_rows=2, spans=((0, 0, 0, 1), (1, 0, 2, 0), (3, 0, 4, 0))), "merged",
            caption="Table 1: Management indicators")
    b.figure("Figure 1: Claims per quarter", "Claims", b.values([120, 95, 140, 110]), ["T1", "T2", "T3", "T4"])
    b.page_break()
    b.heading("3. Appendix: register", 2)
    b.table(TableSpec([["Reference", "Insured", "Amount", "Status"], *_sinistres(b.rng, 40 if b.seed == 0 else b.rng.randint(40, 60), b.seed)], repeat_header=True),
            "multipage")
    return b


# Order matters: with a small ``count`` the first kinds are the richest ones.
KINDS = {
    "constat": doc_accident_report, "rapport": doc_annual_report, "table_nested": doc_table_nested,
    "table_merged": doc_table_merged, "forms": doc_forms, "headings": doc_headings, "lists": doc_lists,
    "table_simple": doc_table_simple, "table_borderless": doc_table_borderless,
    "table_multipage": doc_table_multipage, "numbers": doc_numbers, "links": doc_links, "footnotes": doc_footnotes,
    "figures": doc_figures, "columns": doc_columns, "running": doc_running, "scan": doc_mini,
}


def plan(count: int | None = None, seed: int = 0, only: list[str] | None = None) -> list[tuple[str, int]]:
    """``(kind, seed)`` of every document to write.

    ``count`` documents are taken round-robin over the kinds (one of each when ``count`` is None), so a
    larger count repeats each kind with the next seed: same layout, different content (names, amounts,
    row counts, ticked boxes, filler text).
    """
    kinds = list(only) if only else list(KINDS)
    if bad := [k for k in kinds if k not in KINDS]:
        raise ValueError(f"unknown document kind(s): {', '.join(bad)}. Known: {', '.join(KINDS)}")
    n = len(kinds) if count is None else count
    if n < 1:
        raise ValueError("count must be at least 1")
    return [(kinds[i % len(kinds)], seed + i // len(kinds)) for i in range(n)]


def stem(kind: str, seed: int) -> str:
    return kind if seed == 0 else f"{kind}-s{seed:02d}"


def generate_structured(out_dir: str | Path, *, count: int | None = None, seed: int = 0,
                        only: list[str] | None = None, scan_dpi: int = 150) -> list[Path]:
    """Write ``count`` PDFs with their ``<stem>.gt.json``; returns the PDF paths (see :func:`plan`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, sd in plan(count, seed, only):
        b = KINDS[kind](sd)
        name = stem(kind, sd)
        pdf = out / f"{name}.pdf"
        gt = b.build(pdf)
        gt.name = name
        if b.meta.get("text_layer") is False:
            to_scan(pdf, dpi=scan_dpi, seed=sd)
        gt.save(out / f"{name}.gt.json")
        paths.append(pdf)
    return paths


def check_gt(pdf: Path, gt: GroundTruth) -> list[str]:
    """Problems found when comparing the ground truth with the PDF's text layer (empty = consistent).

    Every text the ground truth asserts must be readable in the PDF, and a footnote must sit on the
    page that cites it. Scans have no text layer and are skipped.
    """
    if gt.meta.get("text_layer") is False:
        return []
    norm = lambda s: " ".join(s.split())  # noqa: E731
    doc = pymupdf.open(pdf)
    pages = [norm(p.get_text()) for p in doc]
    doc.close()
    whole = " ".join(pages)
    problems = []
    for kind, text in gt.texts():
        if kind in ("figure",):
            continue
        if norm(text) not in whole:
            problems.append(f"{kind}: text not found in the PDF: {text[:60]!r}")
    return problems
