"""The structured-extraction generator: PDFs and ground truth must agree."""

import pymupdf
import pytest

from pdf_bench.elements import Element, TYPES, load_gt
from pdf_bench.synth import KINDS, check_gt, generate_structured, plan


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("structured")
    return d, generate_structured(d)


def test_every_document_is_consistent(out):
    _, pdfs = out
    assert len(pdfs) == len(KINDS)
    for pdf in pdfs:
        gt = load_gt(pdf.with_suffix(".gt.json"))
        assert gt.elements, pdf.name
        assert check_gt(pdf, gt) == [], pdf.name
        assert {e.type for e in gt.walk()} <= set(TYPES), pdf.name


def test_category_ground_truth_is_limited_to_its_type(out):
    d, _ = out
    only = {"headings": {"heading", "paragraph"}, "lists": {"list", "list_item"}, "links": {"link"},
            "footnotes": {"footnote"}, "running": {"header", "footer"}, "columns": {"paragraph"}}
    for name, allowed in only.items():
        gt = load_gt(d / f"{name}.gt.json")
        assert {e.type for e in gt.walk()} <= allowed, name


def test_nested_merged_and_multipage_tables(out):
    d, _ = out
    nested = load_gt(d / "table_nested.gt.json")
    assert nested.counts()["table"] == 3  # one outer, two inside cells
    merged = load_gt(d / "table_merged.gt.json")
    spans = [c for c in merged.walk() if c.type == "cell" and ("rowspan" in c.attrs or "colspan" in c.attrs)]
    assert len(spans) == 3
    assert len(pymupdf.open(d / "table_multipage.pdf")) >= 2
    assert load_gt(d / "table_multipage.gt.json").counts()["table"] == 1  # one table over the pages


def test_forms_have_blank_fields_and_ticked_boxes(out):
    gt = load_gt(out[0] / "forms.gt.json")
    fields = [e for e in gt.walk() if e.type == "field"]
    assert any(e.attrs["value"] == "" for e in fields)
    boxes = [e.attrs["checked"] for e in gt.walk() if e.type == "checkbox"]
    assert True in boxes and False in boxes


def test_scan_has_no_text_layer_and_footnotes_sit_on_their_page(out):
    d, _ = out
    assert not pymupdf.open(d / "scan.pdf")[0].get_text().strip()
    assert load_gt(d / "scan.gt.json").meta["text_layer"] is False
    pages = [p.get_text() for p in pymupdf.open(d / "footnotes.pdf")]
    for p in pages:  # a note and the sentence that cites it are on the same page
        assert ("The compensation is calculated" in p) == ("The deductible is set" in p)
        assert ("The payment period" in p) == ("Report written by" in p)


def test_generation_is_reproducible(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_structured(a)
    generate_structured(b)
    for f in a.glob("*.gt.json"):
        assert f.read_text() == (b / f.name).read_text()


def test_element_roundtrip():
    e = Element("table", attrs={"rows": 1}, children=[Element("cell", "x", attrs={"r": 0, "c": 0})])
    assert Element.from_dict(e.to_dict()) == e


def test_count_cycles_over_kinds_with_new_seeds(tmp_path):
    assert plan() == [(k, 0) for k in KINDS]
    assert plan(3) == [(k, 0) for k in list(KINDS)[:3]]
    p = plan(2 * len(KINDS) + 1)
    assert p[len(KINDS)] == (list(KINDS)[0], 1) and p[-1] == (list(KINDS)[0], 2)
    assert plan(4, seed=5, only=["forms", "links"]) == [("forms", 5), ("links", 5), ("forms", 6), ("links", 6)]
    with pytest.raises(ValueError, match="unknown"):
        plan(1, only=["nope"])
    with pytest.raises(ValueError, match="at least 1"):
        plan(0)


def test_variants_differ_in_content_not_in_structure_and_stay_consistent(tmp_path):
    pdfs = generate_structured(tmp_path, count=2 * len(KINDS))
    assert len({p.name for p in pdfs}) == 2 * len(KINDS)
    for pdf in pdfs:
        gt = load_gt(pdf.with_suffix(".gt.json"))
        assert gt.name == pdf.stem
        assert check_gt(pdf, gt) == [], pdf.name
    a, b = load_gt(tmp_path / "forms.gt.json"), load_gt(tmp_path / "forms-s01.gt.json")
    assert a.counts() == b.counts() and a.to_json().replace("forms", "") != b.to_json().replace("forms-s01", "")
    for name in ("forms", "forms-s01"):  # a variant keeps one ticked and one empty box
        boxes = [e.attrs["checked"] for e in load_gt(tmp_path / f"{name}.gt.json").walk() if e.type == "checkbox"]
        assert True in boxes and False in boxes
    assert len(pymupdf.open(tmp_path / "table_multipage-s01.pdf")) >= 2


def test_cli_synth_count(tmp_path, capsys):
    from pdf_bench import cli

    assert cli.main(["synth", str(tmp_path), "--count", "3"]) == 0
    assert len(list(tmp_path.glob("*.pdf"))) == 3
    assert cli.main(["synth", str(tmp_path), "--only", "nope"]) == 2


def test_multipage_variants_always_span_pages(tmp_path):
    for pdf in generate_structured(tmp_path, count=6, only=["table_multipage"]):
        assert len(pymupdf.open(pdf)) >= 2, pdf.name
