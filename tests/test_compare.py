"""Converter, scorer, results cache and the Compare page."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdf_bench import results, selection
from pdf_bench.convert import to_elements
from pdf_bench.elements import load_gt
from pdf_bench.oracle import gt_to_html, gt_to_markdown
from pdf_bench.parsers import REGISTRY, common, databricks
from pdf_bench.score import box_state, norm, score
from pdf_bench.synth import KINDS, generate_structured


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    d = tmp_path_factory.mktemp("docs")
    generate_structured(d)
    return d


def kinds(elements):
    return [(e.type, e.level, e.text) for e in elements]


def test_markdown_conversion():
    md = ("# Title\n\nSome [link](https://x.example/a) here.\n\n## Part\n\n- a\n  - a1\n- b\n\n1. one\n2. two\n\n"
          "| h1 | h2 |\n|---|---|\n| 1 | 2 |\n\n![](pic.png)\n\n[image]\n\n[^1]: A note.\n")
    els = to_elements(md)
    assert kinds(els)[0] == ("heading", 1, "Title")
    para = els[1]
    assert para.type == "paragraph" and para.children[0].attrs["href"] == "https://x.example/a"
    lists = [e for e in els if e.type == "list"]
    assert [i.text for i in lists[0].children] == ["a", "b"] and lists[0].children[0].children[0].children[0].text == "a1"
    assert lists[1].attrs["ordered"] is True
    table = next(e for e in els if e.type == "table")
    assert [(c.attrs["r"], c.attrs["c"], c.text, c.attrs["header"]) for c in table.children][:3] == [
        (0, 0, "h1", True), (0, 1, "h2", True), (1, 0, "1", False)]
    assert sum(e.type == "figure" for e in els) == 2
    assert els[-1].type == "footnote" and els[-1].attrs["marker"] == "1"


def test_html_conversion_spans_and_nesting():
    html = ("<table><tr><th rowspan=2>A</th><th colspan=2>B</th></tr><tr><td>b1</td><td>b2</td></tr>"
            "<tr><td>x</td><td colspan=2><table><tr><td>inner</td></tr></table>outer</td></tr></table>")
    t = to_elements(html)[0]
    cells = {(c.attrs["r"], c.attrs["c"]): c for c in t.children}
    assert cells[(0, 0)].attrs["rowspan"] == 2 and cells[(0, 1)].attrs["colspan"] == 2
    assert cells[(1, 1)].text == "b1"  # the cell under the rowspan is shifted
    assert cells[(2, 1)].text == "outer" and cells[(2, 1)].children[0].children[0].text == "inner"


def test_norm_and_box_state():
    assert norm("**Bold** &amp; <b>x</b>_y") == "bold & x y"
    assert box_state("☒ ") is True and box_state("[ ] ") is False and box_state("plain") is None
    assert box_state("☐ ☒ ") is True  # the mark closest to the label wins


@pytest.mark.parametrize("kind", list(KINDS))
def test_a_perfect_html_answer_scores_one(docs, kind):
    gt = load_gt(docs / f"{kind}.gt.json")
    rep = score(gt, gt_to_html(gt))
    frame = rep.frame()
    assert (frame["structure"].dropna() == 1.0).all(), frame[frame["structure"] < 1].to_dict("records")
    assert (frame["content"].dropna() == 1.0).all()


def test_markdown_ceiling_loses_nested_and_merged_cells(docs):
    nested = score(load_gt(docs / "table_nested.gt.json"), gt_to_markdown(load_gt(docs / "table_nested.gt.json")))
    row = nested.frame().iloc[0]
    assert row["content"] == 1.0 and row["structure"] < 0.75 and "nested" in row["detail"]  # flattened nested cells count half
    merged = score(load_gt(docs / "table_merged.gt.json"), gt_to_markdown(load_gt(docs / "table_merged.gt.json")))
    assert "merged cell" in merged.frame().iloc[0]["detail"]


def test_flat_text_has_content_but_no_structure(docs):
    gt = load_gt(docs / "headings.gt.json")
    flat = "\n".join(e.text for e in gt.elements)
    rep = score(gt, flat)
    ov = rep.overall()
    assert ov["content"] == 1.0 and ov["structure"] < 0.5  # headings lose their level; flat paragraphs still are blocks
    assert "not a heading" in rep.frame().iloc[0]["detail"]


def test_wrong_heading_level_and_lost_link_target(docs):
    gt = load_gt(docs / "headings.gt.json")
    wrong = "\n\n".join("### " + e.text for e in gt.elements)
    f = score(gt, wrong).frame()
    assert (f["structure"] == 0).sum() >= 5 and "expected H" in f.iloc[0]["detail"]
    gl = load_gt(docs / "links.gt.json")
    text = "\n\n".join(f"[{e.text}](https://other.example)" for e in gl.elements)
    assert "target lost" in score(gl, text).frame().iloc[0]["detail"]


def test_checkbox_and_field_scoring():
    from pdf_bench.elements import Element, GroundTruth

    gt = GroundTruth("g", "form", ["field", "checkbox"], [
        Element("field", attrs={"label": "Last name", "value": "Martin"}),
        Element("checkbox", "Injuries", attrs={"checked": False}),
        Element("checkbox", "Report drawn up", attrs={"checked": True})])
    good = score(gt, "| Last name | Martin |\n\n☐ Injuries\n\n☒ Report drawn up").frame()
    assert (good["structure"] == 1).all()
    far = "unrelated words " * 12
    bad = score(gt, f"Last name\n\n{far}\n\nMartin\n\nInjuries\n\nReport drawn up").frame()
    assert bad["structure"].tolist() == [0.0, 0.0, 0.0]


def test_results_cache_and_failure_not_cached(docs, tmp_path, monkeypatch):
    pdf = docs / "headings.pdf"
    first = results.run("oracle_html", pdf, cache_dir=tmp_path)
    assert first["cached"] is False and first["error"] is None and first["pages"]
    assert results.run("oracle_html", pdf, cache_dir=tmp_path)["cached"] is True
    assert results.run("oracle_html", pdf, cache_dir=tmp_path, force=True)["cached"] is False
    bare = tmp_path / "x.pdf"
    bare.write_bytes(pdf.read_bytes())  # no ground truth next to it: the oracle fails
    failed = results.run("oracle_html", bare, cache_dir=tmp_path)
    assert failed["error"] and not list(tmp_path.glob("**/oracle_html/x.*.json"))
    monkeypatch.setenv("PDF_BENCH_CLAUDE_MODEL", "a")
    k1 = results.cache_key(REGISTRY["claude"], pdf)
    monkeypatch.setenv("PDF_BENCH_CLAUDE_MODEL", "b")
    assert results.cache_key(REGISTRY["claude"], pdf) != k1  # changing the model id invalidates the cache


def test_compare_page_end_to_end(docs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sel = selection.Selection(parsers=["pymupdf4llm"], docs_dir=str(docs), docs_selected=["table_nested.pdf", "headings.pdf"])
    sel.save(tmp_path / "b.json")
    page = str(Path(selection.__file__).with_name("page_compare.py"))
    at = AppTest.from_file(page, default_timeout=120)
    at.session_state["selection_file"] = str(tmp_path / "b.json")
    at.run()
    assert not at.exception
    next(b for b in at.button if b.label == "Run the selected models").click()
    at.run()
    assert not at.exception
    table = at.dataframe[0].value  # scores per model, over every PDF of the folder
    assert list(table.index) == ["pymupdf4llm"]  # the references are not parsers: not on this page
    assert table.loc["pymupdf4llm", "PDFs"] == "2/2"  # only the two PDFs of the selection ran
    assert table.loc["pymupdf4llm", "s / PDF"] >= 0
    assert table.loc["pymupdf4llm", "est. cost $"] == 0 and "free" in table.loc["pymupdf4llm", "cost basis"]
    assert any("Recommendation" in m.value for m in at.markdown) and any("Free" in m.value for m in at.markdown)
    next(s for s in at.selectbox if s.label == "PDF").select("table_nested.pdf")
    next(s for s in at.selectbox if s.label == "Model").select_index(0)
    at.run()
    assert not at.exception
    assert {m.label for m in at.metric} >= {"Content", "Structure"}
    assert list((tmp_path / "results").glob("**/pymupdf4llm/*.json"))
    assert not list((tmp_path / "results").glob("**/oracle_html"))


def test_named_variants(docs, monkeypatch):
    pdf = next(docs.glob("headings.pdf"))
    monkeypatch.delenv("PDF_BENCH_CLAUDE_MODEL", raising=False)
    base = results.cache_key(REGISTRY["claude"], pdf)
    with results.settings(REGISTRY["claude"], {"model": "m-big"}):
        big = results.cache_key(REGISTRY["claude"], pdf)
    assert big != base and results.cache_key(REGISTRY["claude"], pdf) == base  # restored afterwards
    sel = selection.Selection(parsers=["pymupdf4llm"], variants={"claude-big": {"parser": "claude", "options": {"model": "m-big"}}})
    assert [e.label for e in sel.entries()] == ["pymupdf4llm", "claude-big"]
    assert sel.entries()[1].parser == "claude" and sel.entries()[1].options == {"model": "m-big"}
    bad = selection.Selection(parsers=["pymupdf4llm"], variants={
        "oracle_html": {"parser": "claude", "options": {}}, "x": {"parser": "claude", "options": {"api_key": "s"}},
        "y": {"parser": "claude", "options": {"nope": "1"}}, "z": {"parser": "ghost", "options": {}}})
    text = " | ".join(bad.problems())
    for needle in ("already a parser name", "looks like a secret", "no setting 'nope'", "unknown parser 'ghost'"):
        assert needle in text


def test_results_are_kept_per_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for ds in ("data/set_a", "data/set_b"):
        d = tmp_path / ds
        d.mkdir(parents=True)
        generate_structured(d, only=["headings"])
        results.run("oracle_html", d / "headings.pdf", cache_dir=tmp_path / "results")
    assert (tmp_path / "results/data/set_a/oracle_html").is_dir() and (tmp_path / "results/data/set_b/oracle_html").is_dir()
    other = tmp_path.parent / "elsewhere"  # outside the working directory: the full path is kept
    other.mkdir(exist_ok=True)
    generate_structured(other, only=["headings"])
    p = results.cache_path(parser=__import__("pdf_bench.parsers", fromlist=["x"]).lookup("oracle_html"),
                           pdf=other / "headings.pdf", cache_dir=Path("results"))
    assert "elsewhere" in p.parts and p.parts[0] == "results" and not p.is_absolute()


def test_analysis_files(tmp_path, monkeypatch):
    from pdf_bench import analysis

    monkeypatch.chdir(tmp_path)
    d = tmp_path / "data" / "set"
    generate_structured(d, only=["forms", "headings"])
    sel = selection.Selection(parsers=["pymupdf4llm"], docs_dir="data/set")
    # one answer on disk for forms, none for headings: the missing one is reported, not run
    results.run("pymupdf4llm", d / "forms.pdf")
    md, js = analysis.write(sel)
    assert md == Path("analysis/data/set/analysis.md") and js.exists()
    import json

    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["missing"] == [{"model": "pymupdf4llm", "pdf": "headings.pdf"}]
    run = data["runs"][0]
    assert run["output"] and run["recovered"] and {"kind", "structure", "evidence"} <= set(run["rows"][0])
    assert "headings.gt" not in md.read_text(encoding="utf-8") and "pymupdf4llm / headings.pdf: no saved answer" in md.read_text(encoding="utf-8")
    # checkboxes are found as text, never marked: flagged as a doubt about the score
    assert any("checkbox" in f and "content 1.0" in f for f in data["flags"])


def test_evidence_ignores_the_row_prefixes():
    from pdf_bench.analysis import evidence

    assert evidence("H2 Circumstances", "# Circumstances\n\ntext")["similarity"] == 100
    assert evidence("Witness: (blank)", "Witness:")["similarity"] == 100


def _gt(*elements):
    from pdf_bench.elements import GroundTruth

    return GroundTruth("t", "composite", [], list(elements))


def test_a_fragment_is_not_the_whole_paragraph():
    from pdf_bench.elements import Element

    para = "Section 6. " + "Compensation appraisal deadline refund premium insured coverage. " * 3 + "Tail of the paragraph here."
    gt = _gt(Element("paragraph", para))
    whole = score(gt, para, parser="x").frame().iloc[0]
    assert whole.structure == 1.0
    cut = score(gt, para[: len(para) // 2] + "\n\nOther block\n\n" + para[len(para) // 2:], parser="x").frame().iloc[0]
    assert cut.structure == 0.0 and "fragment" in cut.detail


def test_list_numbers_are_not_judged():
    from pdf_bench.elements import Element

    items = [Element("list_item", "Declare the claim"), Element("list_item", "Assess the damage")]
    gt = _gt(Element("list", attrs={"ordered": True}, children=items))
    r = score(gt, "- 1 Declare the claim\n- 2 Assess the damage", parser="x").frame().iloc[0]
    assert r.structure == 1.0


def test_footnote_written_as_a_numbered_block_counts():
    from pdf_bench.elements import Element

    gt = _gt(Element("footnote", "The deductible is set in the contract.", attrs={"marker": "1"}))
    for out in ("> 1 The deductible is set in the contract.", "1 The deductible is set in the contract.",
                "- 1 The deductible is set in the contract.", "[^1]: The deductible is set in the contract."):
        assert score(gt, "Body text.\n\n" + out, parser="x").frame().iloc[0].structure == 1.0, out
    plain = score(gt, "Body text. The deductible is set in the contract.", parser="x").frame().iloc[0]
    assert plain.structure == 0.0 and plain.content == 1.0


def test_a_short_block_inside_the_caption_is_not_the_caption():
    from pdf_bench.elements import Element

    gt = _gt(Element("figure", children=[Element("caption", "Figure 1: Claims per quarter")]))
    out = "Table.\n\nClaims\n\nFigure 1: Claims per quarter\n\n<!-- image -->\n\nNext."
    assert score(gt, out, parser="x").frame().iloc[0].structure == 1.0


def test_databricks_footnotes_are_split_per_line():

    parsed = {"document": {"pages": [{"id": 0}], "elements": [
        {"id": 0, "type": "footnote", "content": "1 First note.\n2 Second note.", "bbox": [{"page_id": 0}]}]}}
    assert databricks.databricks_elements_to_pages(parsed) == ["[^1]: First note.\n\n[^2]: Second note."]


def test_compare_with_steps_shows_raw_and_gain(docs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sel = selection.Selection(parsers=["pymupdf4llm"], docs_dir=str(docs), docs_selected=["table_multipage.pdf", "lists.pdf"],
                              postprocess=["running_text", "merge_tables", "lists"])
    sel.save(tmp_path / "b.json")
    for name in ("table_multipage.pdf", "lists.pdf"):
        results.run("pymupdf4llm", docs / name)
    base = str(Path(selection.__file__).parent)
    at2 = AppTest.from_file(f"{base}/page_compare.py", default_timeout=120)
    at2.session_state["selection_file"] = str(tmp_path / "b.json")
    at2.run()
    assert not at2.exception
    assert any("post-processing: running text" in c.value.lower() or "post-processing:" in c.value for c in at2.caption)
    cols = at2.dataframe[0].value.columns
    assert "raw structure" in cols and "post-processing gain" in cols


def test_pymupdf_picture_comments_are_a_figure_next_to_the_caption():
    from pdf_bench.elements import Element

    out = "Text.\n\n<!-- Start of picture text -->\nClaims re ported<br>T1 T2<br><!-- End of picture text -->\n\n_Figure 1: Claims_ \n\nMore."
    gt = _gt(Element("figure", children=[Element("caption", "Figure 1: Claims")]))
    assert score(gt, out, parser="x").frame().iloc[0].structure == 1.0


def test_numbers_must_be_exact():
    from pdf_bench.elements import Element

    gt = _gt(Element("paragraph", "Paid 1,234,567.89 EUR on 12/03/2025 at 5:45 pm."))
    ok = score(gt, "Paid 1,234,567.89 EUR on 12/03/2025 at 5:45 pm.", parser="x").frame()
    row = ok[ok.kind == "numbers"].iloc[0]
    assert row.content == 1.0 and row.detail == ""
    bad = score(gt, "Paid 1,234,767.89 EUR on 12/08/2025 at 5:45 pm.", parser="x").frame()
    row = bad[bad.kind == "numbers"].iloc[0]
    assert abs(row.content - 1 / 3) < 1e-9 and "1,234,567.89" in row.detail and "12/03/2025" in row.detail  # the text itself is still "found"
    assert bad[bad.kind == "paragraph"].iloc[0].content == 1.0


def test_title_row():
    from pdf_bench.elements import Element

    gt = _gt(Element("heading", "Annual report", level=1), Element("heading", "Results", level=2))
    t = lambda out: score(gt, out, parser="x").frame().query("kind == 'title'").iloc[0]  # noqa: E731
    assert t("# Annual report\n\n## Results").structure == 1.0
    r = t("## Annual report\n\n## Results")
    assert r.structure == 0.5 and "H2" in r.detail
    assert t("## Results\n\n# Annual report").structure == 0.0
    assert t("Annual report\n\n## Results").structure == 0.0


def test_section_path_follows_the_heading_levels():
    from pdf_bench.elements import Element

    table = Element("table", children=[Element("cell", "Reference", attrs={"r": 0, "c": 0}),
                                       Element("cell", "Amount", attrs={"r": 0, "c": 1})])
    gt = _gt(Element("heading", "Report", level=1), Element("heading", "2. Method", level=2),
             Element("heading", "2.1 Data", level=3), table)
    md = "| Reference | Amount |\\n|---|---|\\n| 1 | 2 |"
    good = score(gt, "# Report\n\n## 2. Method\n\n### 2.1 Data\n\n" + md.replace("\\n", "\n"), parser="x").frame()
    assert good[good.kind == "section_path"].iloc[0].structure == 1.0
    flat = score(gt, "# Report\n\n## 2. Method\n\n## 2.1 Data\n\n" + md.replace("\\n", "\n"), parser="x").frame()
    row = flat[flat.kind == "section_path"].iloc[0]
    assert row.structure == 0.0 and "Report > 2.1 Data" in row.detail  # flat levels lose the parent: a wrong chunk context


def test_page_of_each_element(tmp_path):
    from pdf_bench import locate, view
    from pdf_bench.elements import load_gt

    pdf = generate_structured(tmp_path, only=["rapport"])[0]  # three pages
    gt = locate.annotate(load_gt(pdf.with_suffix(".gt.json")), pdf)
    placed = [(e, e.attrs["page"]) for e in gt.elements if e.attrs.get("page")]
    assert len({p for _, p in placed}) >= 2
    per_page = [[] for _ in range(max(p for _, p in placed))]
    for e, page in placed:
        text = next((c.text for c in e.children if c.type == "caption"), "") if e.type == "figure" else e.text
        per_page[page - 1].append(text)
    right = view.frame("x", "pymupdf4llm", pdf, ["\n\n".join(p) for p in per_page])
    assert right[right.kind == "page"].iloc[0].structure == 1.0
    joined = view.frame("x", "pymupdf4llm", pdf, ["\n\n".join("\n\n".join(p) for p in per_page)])
    row = joined[joined.kind == "page"].iloc[0]
    assert row.structure == 0.0 and "not split by page" in row.detail
    shifted = view.frame("x", "pymupdf4llm", pdf, list(reversed(["\n\n".join(p) for p in per_page])))
    assert shifted[shifted.kind == "page"].iloc[0].structure < 1.0


def test_numbers_with_space_thousands_is_one_value(tmp_path):
    from pdf_bench.score import _numbers
    assert _numbers("paid 12 480,50 EUR on 03/11/2025 at 14:35") == ["12480,50", "03/11/2025", "14:35"]
    assert _numbers("31 December 2025 640") == ["31", "2025", "640"]  # not a thousands group


def test_numbers_document_discriminates(tmp_path):
    from pdf_bench.synth import check_gt, generate_structured
    generate_structured(tmp_path, only=["numbers"])
    gt = load_gt(tmp_path / "numbers.gt.json")
    assert check_gt(tmp_path / "numbers.pdf", gt) == []
    txt = "\n".join(t for _, t in gt.texts())
    assert score(gt, txt).frame().query("kind == 'numbers'").iloc[0]["content"] == 1.0
    bad = txt.replace("12 480,50", "12 408,50").replace("14:35", "14.35")
    assert score(gt, bad).frame().query("kind == 'numbers'").iloc[0]["content"] < 1.0


def test_checkbox_state_written_after_the_label():
    from pdf_bench.elements import Element, GroundTruth
    gt = GroundTruth("c", "form", ["checkbox"], [Element("checkbox", "Injuries were reported", attrs={"checked": True}),
                                   Element("checkbox", "Vehicle is immobilised", attrs={"checked": False})])
    rows = score(gt, "option Injuries were reported: [x]\n\noption Vehicle is immobilised: [ ]").frame()
    assert list(rows[rows.kind == "checkbox"]["structure"]) == [1.0, 1.0]
    rows = score(gt, "[ ] Injuries were reported\n\n[ ] Vehicle is immobilised").frame()
    assert list(rows[rows.kind == "checkbox"]["structure"]) == [0.0, 1.0]


def test_div_captions_and_bracketed_figure_markers_are_blocks():
    from pdf_bench.convert import to_elements
    els = to_elements('<table><tr><td>a</td></tr></table>\n\n<div style="text-align: center;">Figure 1: Claims</div>\n\n[Figure: Bar chart of claims]')
    assert [e.type for e in els] == ["table", "paragraph", "figure"] and els[1].text == "Figure 1: Claims"


def test_chart_read_as_data_counts_as_a_figure_and_footnote_lines_are_split():
    from pdf_bench.elements import Element, GroundTruth
    fig = Element("figure", attrs={"image_text": ["Claims", "T1", "T2"]}, children=[Element("caption", "Figure 1: Claims per quarter")])
    note = [Element("footnote", "The deductible is set.", attrs={"marker": "1"}), Element("footnote", "Written by an expert.", attrs={"marker": "2"})]
    gt = GroundTruth("g", "x", ["figure", "footnote"], [fig, *note])
    out = "| Claims | n |\n|---|---|\n| T1 | 1 |\n| T2 | 2 |\n\n*Figure 1: Claims per quarter*\n\n1 The deductible is set.\n2 Written by an expert."
    rows = score(gt, out).frame()
    assert list(rows[rows.kind == "figure"]["structure"]) == [1.0]
    assert list(rows[rows.kind == "footnote"]["structure"]) == [1.0, 1.0]
    rows = score(gt, "Claims\n\nFigure 1: Claims per quarter").frame()
    assert list(rows[rows.kind == "figure"]["structure"]) == [0.0]  # the labels of the chart are missing


def test_incomplete_page_is_an_error_and_image_models_cannot_return_links():
    import pytest

    from pdf_bench.parsers import REGISTRY
    with pytest.raises(RuntimeError, match="incomplete"):
        common.check_complete("max_tokens", ("end_turn", None), "stop_reason", 2)
    assert all("link" in REGISTRY[n].cannot for n in ("claude", "gemini", "gpt"))
