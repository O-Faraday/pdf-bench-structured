"""Format-free model of what a PDF contains: a tree of typed elements.

The ground truth of the structured benchmark is written in this model, and (next step) every parser
output, whatever its format, is converted to it before scoring. Tests therefore never look at syntax.

Element types
-------------
Text        ``heading`` (level), ``paragraph``, ``list`` (ordered) > ``list_item`` (nested ``list``),
            ``link`` (href), ``footnote`` (marker, text)
Tables      ``table`` > ``cell`` (r, c, rowspan, colspan, header); a cell may hold a nested ``table``
            or a ``checkbox``
Forms       ``field`` (label, value; the value may be empty), ``checkbox`` (label, checked)
Media       ``figure`` > ``caption``
Page        ``header``, ``footer`` (repeated text that should not be part of the body)

In a ground-truth file ``elements`` lists, in expected reading order, **only the elements of the
types in ``tested``**: the document may contain other content, which is not asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = 2

TYPES: dict[str, str] = {
    "heading": "A title; `level` 1 is the highest.",
    "paragraph": "A block of running text.",
    "list": "A bulleted or numbered list (`ordered`); its children are `list_item`s.",
    "list_item": "One item of a list; a nested list is a child.",
    "link": "Clickable text; `href` is its target.",
    "footnote": "Text at the foot of the page, tied to a `marker` in the body.",
    "table": "A grid of `cell`s; may be nested inside a cell, may span pages.",
    "cell": "One table cell at row `r`, column `c`; `header` for a header cell; may span rows or columns.",
    "field": "A form field: `label` and `value` (a blank field has an empty value).",
    "checkbox": "A box that is ticked or not (`checked`), with its label.",
    "figure": "An image; its `caption` is a child.",
    "caption": "The legend of a figure or table.",
    "header": "Running text repeated at the top of every page.",
    "footer": "Running text repeated at the bottom of every page.",
}


@dataclass
class Element:
    type: str
    text: str = ""
    level: int | None = None
    attrs: dict = field(default_factory=dict)
    children: list["Element"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.text:
            d["text"] = self.text
        if self.level is not None:
            d["level"] = self.level
        if self.attrs:
            d["attrs"] = self.attrs
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Element":
        return cls(d["type"], d.get("text", ""), d.get("level"), dict(d.get("attrs", {})),
                   [cls.from_dict(c) for c in d.get("children", [])])

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class GroundTruth:
    name: str
    category: str  # what the document is meant to test (a single element kind, or "composite")
    tested: list[str]  # element types asserted
    elements: list[Element]
    meta: dict = field(default_factory=dict)  # e.g. text_layer=False for scans

    def walk(self):
        for e in self.elements:
            yield from e.walk()

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.walk():
            out[e.type] = out.get(e.type, 0) + 1
        return out

    def texts(self) -> list[tuple[str, str]]:
        """(element type, text) of everything that must be readable in the PDF text layer."""
        out = []
        for e in self.walk():
            for key in ("text",):
                if getattr(e, key):
                    out.append((e.type, e.text))
            for key in ("label", "value", "marker"):
                if e.attrs.get(key):
                    out.append((e.type, str(e.attrs[key])))
        return out

    def to_json(self) -> str:
        return json.dumps({"schema": SCHEMA, "name": self.name, "category": self.category,
                           "tested": self.tested, "meta": self.meta, "ordered": True,
                           "elements": [e.to_dict() for e in self.elements]},
                          indent=1, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def load_gt(path: str | Path) -> GroundTruth:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{path}: ground-truth schema {d.get('schema')} (expected {SCHEMA})")
    return GroundTruth(d["name"], d["category"], d["tested"], [Element.from_dict(e) for e in d["elements"]],
                       d.get("meta", {}))
