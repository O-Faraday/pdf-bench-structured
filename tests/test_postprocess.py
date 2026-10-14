from pdf_bench import postprocess as pp
from pdf_bench.convert import to_elements

MD1 = "| Ref | Name |\n|---|---|\n| 1 | A |\n| 2 | B |"
MD2 = "| Ref | Name |\n|---|---|\n| 3 | C |"


def _rows(text):
    t = next(e for e in to_elements(text) if e.type == "table")
    return max(c.attrs["r"] for c in t.children) + 1


def test_merge_markdown_tables_with_repeated_header():
    out = pp.apply([MD1, MD2], ["merge_tables"])
    assert len(out) == 1 and out[0].count("| Ref") == 1 and _rows(out[0]) == 4  # header + 3 rows


def test_merge_across_a_page_break_without_repeated_header():
    a = "| Ref | Name |\n|---|---|\n| 1 | A |"
    b = "| 2 | B |\n|---|---|\n| 3 | C |"
    out = pp.apply([a, b], ["merge_tables"])[0]
    assert _rows(out) == 4 and out.count("|---|") == 1  # header + 3 rows


def test_two_real_tables_on_one_page_are_left_alone():
    one = "| A | B |\n|---|---|\n| 1 | 2 |\n\ntext\n\n| X | Y |\n|---|---|\n| 3 | 4 |"
    assert pp.apply([one], ["merge_tables"])[0].count("|---|") == 2
    different = "| A | B |\n|---|---|\n| 1 | 2 |\n\n| X | Y | Z |\n|---|---|---|\n| 3 | 4 | 5 |"
    assert pp.apply([different], ["merge_tables"])[0].count("|---|") >= 1 and "| Z |" in pp.apply([different], ["merge_tables"])[0]  # different width, no repeated header


def test_merge_html_tables():
    a = "<table><tr><th>Ref</th><th>Name</th></tr><tr><td>1</td><td>A</td></tr></table>"
    b = "<table><tr><th>Ref</th><th>Name</th></tr><tr><td>2</td><td>B</td></tr></table>"
    out = pp.apply([a, b], ["merge_tables"])[0]
    assert out.count("<table") == 1 and out.count("<tr>") == 3 and _rows(out) == 3


def test_plain_lines_become_nested_lists():
    plain = "1 Declare the claim\n2 Have the damage assessed\n1 Contact the expert\n2 Keep the items\n3 Send the documents\n4 Wait"
    out = pp.apply([plain], ["lists"])[0]
    assert out.split("\n") == ["- Declare the claim", "- Have the damage assessed", "   - Contact the expert",
                               "   - Keep the items", "- Send the documents", "- Wait"]
    lst = next(e for e in to_elements(out) if e.type == "list")
    assert [c.text for c in lst.children] == ["Declare the claim", "Have the damage assessed", "Send the documents", "Wait"]
    assert pp.apply(["• one\n• two"], ["lists"])[0] == "- one\n- two"


def test_lists_leaves_other_text_alone():
    for text in ("2025 report\n2026 plan", "Just a line", "- already\n- a list", "1 alone", "3 Third\n4 Fourth"):
        assert pp.apply([text], ["lists"])[0] == text


def test_heading_levels_from_section_numbers():
    md = "# Claims report\n\n## 1. Method\n\n## 1.1 Motor policies\n\n## Conclusion"
    out = pp.apply([md], ["heading_levels"])[0]
    assert out.split("\n\n") == ["# Claims report", "## 1. Method", "### 1.1 Motor policies", "## Conclusion"]
    assert pp.apply(["<h2>1.1.1 Deep</h2>"], ["heading_levels"])[0] == "<h4>1.1.1 Deep</h4>"


def test_unknown_steps_are_ignored_and_input_is_not_changed():
    pages = ["a\n\nb"]
    assert pp.apply(pages, ["nope"]) == pages and pages == ["a\n\nb"]


def test_running_text_is_dropped_only_when_repeated_over_pages():
    pages = ["ACME - File 2025\n\nBody one.\n\nConfidential - page 1", "ACME - File 2025\n\nBody two.\n\nConfidential - page 2"]
    assert pp.apply(pages, ["running_text"]) == ["Body one.", "Body two."]
    one = ["ACME - File 2025\n\nBody one.\n\nConfidential - page 1"]
    assert pp.apply(one, ["running_text"]) == one  # one page: nothing repeats


def test_running_text_then_merge_joins_tables_around_page_furniture():
    p1 = "ACME\n\n| Ref | Name |\n|---|---|\n| 1 | A |\n\nPage 1"
    p2 = "ACME\n\n| Ref | Name |\n|---|---|\n| 2 | B |\n\nPage 2"
    out = pp.apply([p1, p2], ["running_text", "merge_tables"])[0]
    assert out.count("|---|") == 1 and out.count("| Ref") == 1 and "ACME" not in out and "Page" not in out


def test_a_tool_that_cannot_express_lists_is_scored_on_them_once_a_step_builds_them(tmp_path, monkeypatch):
    from pdf_bench import view
    from pdf_bench.synth import generate_structured

    pdf = generate_structured(tmp_path, only=["lists"])[0]
    raw = view.frame("databricks_ai_parse", "databricks_ai_parse", pdf, ["1 Declare\n2 Assess"], ())
    assert raw[raw.kind == "list"]["structure"].isna().all() and "not applicable" in raw[raw.kind == "list"].iloc[0].detail
    built = view.frame("databricks_ai_parse", "databricks_ai_parse", pdf, ["1 Declare\n2 Assess"], ("lists",))
    assert built[built.kind == "list"]["structure"].notna().all()
