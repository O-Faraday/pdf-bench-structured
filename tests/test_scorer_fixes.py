"""Fixes to the scorer, converter and post-processing found by comparing the rules with the LLM judge."""

from pdf_bench import postprocess
from pdf_bench.convert import to_elements
from pdf_bench.elements import Element, GroundTruth
from pdf_bench.score import box_state, chunk_coverage, level_offset, norm, score


def gt(*elements):
    return GroundTruth("t", "composite", sorted({e.type for e in elements}), list(elements), {})


def row(g, output, kind):
    return next(r for r in score(g, output, parser="x").rows if r.kind == kind)


def cell(r, c, text="", **kw):
    return Element("cell", text, attrs={"r": r, "c": c, **kw})


def test_html_table_with_blank_lines_keeps_its_nested_tables():
    html = ("<table>\n<tr><th>A</th><th>B</th></tr>\n<tr><td>\n\n<table>\n<tr><td>x</td><td>1</td></tr>\n</table>\n\n</td>"
            "<td>\n\n<table>\n<tr><td>y</td><td>2</td></tr>\n</table>\n\n</td></tr>\n</table>\n\nAfter.")
    tables = [e for e in to_elements(html) if e.type == "table"]
    assert len(tables) == 1 and sum(1 for c in tables[0].children if any(k.type == "table" for k in c.children)) == 2


def test_flattened_nested_table_gets_half_credit():
    inner = Element("table", children=[cell(0, 0, "x"), cell(0, 1, "1")], attrs={"rows": 1, "cols": 2})
    outer = Element("table", children=[cell(0, 0, "A"), Element("cell", "", attrs={"r": 0, "c": 1}, children=[inner])],
                    attrs={"rows": 1, "cols": 2})
    flat = "| A | x | 1 |\n|---|---|---|\n"
    r = row(gt(outer), flat, "table")
    assert 0.4 < r.structure < 0.9 and "flattened ones count half" in r.detail


def test_bullet_glyph_lines_one_per_block_become_a_list():
    out = postprocess.lists(["# T\n\n• one\n\n• two\n\n• three\n\nText after."])[0]
    assert "- one\n- two\n- three" in out and out.endswith("Text after.")


def test_numbers_closing_a_page_stay_footnotes():
    page = "Body text.\n\n1 First note.\n\n2 Second note."
    assert postprocess.lists([page])[0] == page
    inside = "Steps\n\n1 Do this\n\n2 Do that\n\nMore text after the steps."
    assert "- Do this\n- Do that" in postprocess.lists([inside])[0]


def test_dash_numbers_are_nested_by_their_restart():
    out = postprocess.lists(["- 1 First\n- 2 Second\n- 1 Inner one\n- 2 Inner two\n- 3 Third"])[0]
    assert out.splitlines()[2] == "   - Inner one" and out.splitlines()[-1] == "- Third"


def test_more_checkbox_marks():
    assert box_state("✗ done") is True and box_state("$\\times$") is True and box_state("$\\square$") is False


def test_link_may_wrap_more_words():
    g = gt(Element("link", "the reference document", attrs={"href": "https://a.example/r"}))
    assert row(g, "See [Details are in the reference document.](https://a.example/r)", "link").structure == 1.0
    assert row(g, "See [Details are in the reference document.](https://other.example)", "link").structure == 0.0


def test_constant_level_shift_is_a_right_hierarchy():
    heads = [Element("heading", "Report", level=1), Element("heading", "Part A", level=2),
             Element("heading", "Part B", level=2), Element("heading", "Deep", level=3)]
    shifted = "## Report\n\n### Part A\n\n### Part B\n\n#### Deep\n\ntext"
    r = score(gt(*heads), shifted, parser="x").rows
    assert [x.structure for x in r if x.kind == "heading"] == [1.0] * 4
    assert "shifted by +1" in next(x.detail for x in r if x.kind == "heading")
    assert next(x for x in r if x.kind == "title").structure == 1.0
    # all at one level, or one heading off: no common shift, so no credit
    flat = "## Report\n\n## Part A\n\n## Part B\n\n## Deep"
    assert [x.structure for x in score(gt(*heads), flat, parser="x").rows if x.kind == "heading"] != [1.0] * 4
    assert level_offset(heads[:2], to_elements(shifted)) == 0  # two headings are not enough evidence


def test_field_value_must_come_before_the_next_label():
    g = gt(Element("field", attrs={"label": "Last name", "value": "Martin"}), Element("field", attrs={"label": "First name", "value": "Claire"}))
    assert [r.structure for r in score(g, "Last name\n\nFirst name Martin Claire", parser="x").rows if r.kind == "field"] == [0.0, 1.0]
    assert [r.structure for r in score(g, "Last name Martin\n\nFirst name Claire", parser="x").rows if r.kind == "field"] == [1.0, 1.0]


def test_figure_caption_in_a_list_item_and_chart_read_as_data():
    g = gt(Element("figure", attrs={"image_text": ["Claims reported", "T1", "T2", "T3", "T4"]},
                   children=[Element("caption", "Figure 1: Claims per quarter")]))
    assert row(g, "![](img.png)\n\n- Figure 1: Claims per quarter", "figure").structure == 1.0
    chart = "<table><tr><th>T1</th><td>1</td></tr><tr><th>T2</th><td>2</td></tr><tr><th>T3</th><td>3</td></tr><tr><th>T4</th><td>4</td></tr></table>\n\n*Figure 1: Claims per quarter*"
    assert row(g, chart, "figure").structure == 1.0  # four labels out of five
    assert row(g, "Some text.\n\n*Figure 1: Claims per quarter*", "figure").structure == 0.0


def test_section_path_counts_a_block_written_as_another_kind():
    g = gt(Element("heading", "Notice", level=1), Element("list", attrs={"ordered": False}, children=[
        Element("list_item", "Annual premium: 640 EUR"), Element("list_item", "Deductible: 150 EUR")]))
    r = next(x for x in score(g, "# Notice\n\nAnnual premium: 640 EUR Deductible: 150 EUR", parser="x").rows if x.kind == "section_path")
    assert r.structure == 1.0


def test_a_paragraph_cut_in_two_blocks_keeps_most_of_its_content():
    text = " ".join(f"word{i}x" for i in range(30))
    half = " ".join(f"word{i}x" for i in range(18))
    rest = " ".join(f"word{i}x" for i in range(18, 30))
    hay = norm(half + " Another section. " + rest)
    assert chunk_coverage(text, hay) >= 0.8
    assert chunk_coverage(text, norm("nothing in common here at all")) == 0.0


def test_rows_shifted_by_a_merged_header_get_half_credit():
    cells = [cell(0, 0, "Area", header=True, rowspan=2), cell(0, 1, "Plan", header=True, colspan=2),
             cell(1, 1, "Monthly", header=True), cell(1, 2, "Annual", header=True),
             cell(2, 0, "North"), cell(2, 1, "32"), cell(2, 2, "360"), cell(3, 0, "South"), cell(3, 1, "29"), cell(3, 2, "330")]
    table = Element("table", children=cells, attrs={"rows": 4, "cols": 3})
    one_header_row = "| Area | Plan Monthly | Plan Annual |\n|---|---|---|\n| North | 32 | 360 |\n| South | 29 | 330 |\n"
    r = row(gt(table), one_header_row, "table")
    assert 0.2 < r.structure < 0.7 and "rows shifted" in r.detail


def test_a_big_table_with_the_same_letters_is_not_a_chart():
    g = gt(Element("figure", attrs={"image_text": ["Sketch", "A", "B"]}, children=[Element("caption", "Sketch of the accident")]))
    big = ("<table><tr><th>Circumstance</th><th>A</th><th>B</th></tr>"
           + "<tr><td>Was parked in the street</td><td>x</td><td>y</td></tr>" * 4 + "</table>")
    assert row(g, big + "\n\nSketch\n\nSketch of the accident", "figure").structure == 0.0
    small = "Sketch\n\n<table><tr><th>A</th><td>3</td></tr><tr><th>B</th><td>2</td></tr></table>\n\nSketch of the accident"
    assert row(g, small, "figure").structure == 1.0


def test_short_texts_must_be_whole_words():
    g = gt(Element("heading", "Signatures", level=2))
    r = row(g, "Driver A\n\n[signature]", "heading")
    assert r.content == 0.0 and r.structure == 0.0
    assert row(g, "## Signatures", "heading").content == 1.0


def test_a_cell_in_the_right_place_with_its_merge_lost_gets_half():
    merged = Element("table", children=[cell(0, 0, "Plan", header=True, colspan=2), cell(1, 0, "Monthly"), cell(1, 1, "Annual")],
                     attrs={"rows": 2, "cols": 2})
    flat = "| Plan | extra |\n|---|---|\n| Monthly | Annual |\n"
    r = row(gt(merged), flat, "table")
    assert 0.7 < r.structure < 1.0 and "merged cell" in r.detail


def test_bullet_glyph_lines_inside_a_paragraph_block_become_a_list():
    out = postprocess.lists(["## Notice\n\nYour policy expires soon.\n• Annual premium\n• Deductible\n• Roadside assistance"])[0]
    assert "Your policy expires soon.\n\n- Annual premium\n- Deductible\n- Roadside assistance" in out
    assert postprocess.lists(["Intro.\n\n• Only one item\n\nText."])[0] == "Intro.\n\n- Only one item\n\nText."
    table = "<table><tr><td>• x</td></tr>\n</table>"
    assert postprocess.lists([table])[0] == table


def test_reference_model_is_never_ranked():
    from pdf_bench import recommend

    base = {"free": False, "cost": 1.0, "seconds": 1, "runs": 3, "expected_runs": 3, "errors": 0, "not_applicable": [],
            "overall": 0.9, "ocr": None, "weak": []}
    out = recommend.advise([{**base, "name": "claude", "focus": 1.0}, {**base, "name": "gemini", "focus": 0.9}])
    assert [m["name"] for m in out["paid"]] == ["gemini"]
    assert any(n == "claude" and "reference" in why for n, why in out["excluded"])


def test_add_odr_saves_variants(tmp_path):
    from pdf_bench import cli, selection

    path = tmp_path / "b.json"
    assert cli.main(["add-odr", "openai/gpt-5.6-terra", "mistral/mistral-ocr-4-1-annotations", "--selection", str(path)]) == 0
    sel = selection.load(path)
    assert sel.variants["odr:gpt-5.6-terra"] == {"parser": "opendocrouter", "options": {"model": "openai/gpt-5.6-terra"}}
    assert "odr:mistral-ocr-4-1-annotations" in sel.variants and sel.allow_hosted
