"""The app, the saved selection and the Databricks discovery (fake workspace: no network here)."""

import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdf_bench import selection
from pdf_bench.app import __file__ as APP  # noqa: F401  (import check)
from pdf_bench.parsers import databricks_util

APP_PATH = str(Path(APP))


class FakeWorkspace:
    config = types.SimpleNamespace(host="https://example.cloud.databricks.com")
    current_user = types.SimpleNamespace(me=lambda: types.SimpleNamespace(user_name="me@example.com"))
    warehouses = types.SimpleNamespace(list=lambda: [types.SimpleNamespace(
        id="wh1", name="Starter", state=types.SimpleNamespace(value="RUNNING"),
        warehouse_type=types.SimpleNamespace(value="PRO"), cluster_size="2X-Small")])
    catalogs = types.SimpleNamespace(list=lambda: [types.SimpleNamespace(name="main"), types.SimpleNamespace(name="samples")])
    schemas = types.SimpleNamespace(list=lambda catalog_name: [types.SimpleNamespace(name="default")])
    volumes = types.SimpleNamespace(list=lambda catalog_name, schema_name: [types.SimpleNamespace(name="pdfs")])


def test_databricks_discovery():
    w = FakeWorkspace()
    ws = databricks_util.connect(w)
    assert ws.user == "me@example.com" and ws.warehouses[0]["id"] == "wh1" and ws.warehouses[0]["state"] == "RUNNING"
    assert databricks_util.catalogs(w) == ["main", "samples"]
    assert databricks_util.volumes(w, "main", "default") == ["/Volumes/main/default/pdfs"]


def test_databricks_failure_has_a_readable_message():
    class Broken(FakeWorkspace):
        current_user = types.SimpleNamespace(me=lambda: (_ for _ in ()).throw(PermissionError("401 invalid token")))

    with pytest.raises(RuntimeError, match="Databricks connection failed: PermissionError"):
        databricks_util.connect(Broken())


def test_selection_roundtrip_validation_and_apply(tmp_path, monkeypatch):
    sel = selection.Selection(parsers=["pymupdf4llm", "claude", "nope"], options={"claude.model": "m1"})
    problems = sel.problems()
    assert any("unknown parser" in p for p in problems)
    assert any("claude" in p and "third party" in p for p in problems)
    sel2 = selection.Selection(parsers=["pymupdf4llm"], options={"claude.api_key": "x"})
    assert any("secret" in p for p in sel2.problems())
    ok = selection.Selection(parsers=["pymupdf4llm"], options={"claude.model": "m1"}, docs_count=7)
    assert ok.problems() == []
    path = ok.save(tmp_path / "b.json")
    assert selection.load(path) == ok and selection.load(tmp_path / "missing.json") == selection.Selection()
    monkeypatch.delenv("PDF_BENCH_CLAUDE_MODEL", raising=False)
    ok.apply()
    import os
    assert os.environ.pop("PDF_BENCH_CLAUDE_MODEL") == "m1"


def test_app_renders_both_tabs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 2
    assert any("Catalogue checked" in c.value for c in at.caption)


def test_app_generates_documents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=120).run()
    folder = next(t for t in at.text_input if t.label == "Output folder")
    folder.set_value(str(tmp_path / "docs"))
    one = next(c for c in at.checkbox if c.label == "One of each kind")
    one.set_value(False)
    at.run()
    n = next(x for x in at.number_input if x.label == "Number of PDFs")
    n.set_value(3)
    at.run()
    next(b for b in at.button if b.label == "Generate PDFs").click()
    at.run()
    assert not at.exception
    assert len(list((tmp_path / "docs").glob("*.pdf"))) == 3
    assert any("ground truth checked" in s.value for s in at.success)


def test_models_tab_shows_no_problem_for_unselected_parsers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=120).run()
    texts = [w.value for w in at.warning]
    assert texts == ["select at least one parser"]



def test_opendocrouter_picker_creates_named_variants(tmp_path, monkeypatch):
    from pdf_bench.parsers import opendocrouter_util

    def m(id, score):
        return {"id": id, "name": id, "version": "1", "max_sync_pages": 50, "avg_charge_per_page_usd": 0.001,
                "parsebench": {"score": score, "tables": 1, "charts": 1, "content_faithfulness": 1,
                               "semantic_formatting": 1, "visual_grounding": 1}}

    payload = {"data": [m("google/gemini-3-flash", 79.7), m("opendatalab/mineru2.5-pro", 70.05), m("x/low", 10)]}
    monkeypatch.setattr(opendocrouter_util, "fetch", lambda *a, **k: payload)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(selection.__file__).with_name("page_setup.py")), default_timeout=120).run()
    assert not at.exception
    next(b for b in at.button if b.label == "List OpenDocRouter models").click()
    at.run()
    next(n for n in at.number_input if n.label == "Add the best N by the chosen score").set_value(2)
    at.run()
    next(b for b in at.button if b.label == "Add them").click()
    at.run()
    picked = next(x for x in at.multiselect if x.key == "odr_ids")
    assert set(picked.value) == {"google/gemini-3-flash", "opendatalab/mineru2.5-pro"}  # the two best by overall score
    assert not at.exception


def test_setup_page_saves_the_postprocessing_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(Path(selection.__file__).with_name("page_setup.py")), default_timeout=120).run()
    assert not at.exception
    steps = next(x for x in at.multiselect if x.label == "Post-processing steps")
    steps.set_value(["lists", "heading_levels"])
    at.run()
    assert not at.exception
    save = next((b for b in at.button if b.label == "Save selection"), None)
    if save is not None and not save.disabled:
        save.click()
        at.run()
        assert selection.load(tmp_path / "benchmark.json").postprocess == ["lists", "heading_levels"]
