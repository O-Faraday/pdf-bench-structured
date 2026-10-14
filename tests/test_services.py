"""LlamaParse, Unstructured and OpenDocRouter adapters and the model list, with fakes (no network)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pdf_bench.convert import to_elements
from pdf_bench.parsers import (
    REGISTRY,
    llamaparse,
    opendataloader,
    opendocrouter,
    opendocrouter_util,
    unstructured,
)

PDF = Path(__file__).with_name("x.pdf")


@pytest.fixture()
def pdf(tmp_path):
    import pymupdf

    p = tmp_path / "a.pdf"
    d = pymupdf.open()
    d.new_page().insert_text((72, 72), "hello")
    d.save(p)
    return p


def kinds(text):
    return [(e.type, e.level) for e in to_elements(text)]


# --- LlamaParse ----------------------------------------------------------------------------------------------------

def test_llamaparse_adapter(pdf, monkeypatch):
    import llama_cloud

    seen = {}

    class Files:
        def create(self, file, purpose):
            seen["file"], seen["purpose"] = file, purpose
            return SimpleNamespace(id="f1")

    class Parsing:
        def parse(self, **kw):
            seen["parse"] = kw
            pages = [SimpleNamespace(markdown="# One"), SimpleNamespace(markdown="Two")]
            return SimpleNamespace(markdown=SimpleNamespace(pages=pages), model_dump=lambda: {"ok": 1})

    class Client:
        def __init__(self, api_key=None):
            seen["key"] = api_key
            self.files, self.parsing = Files(), Parsing()

    monkeypatch.setattr(llama_cloud, "LlamaCloud", Client)
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "llx")
    pages = llamaparse.llamaparse_pages(REGISTRY["llamaparse"], pdf)
    assert list(pages) == ["# One", "Two"] and pages.raw == {"ok": 1}
    assert seen["key"] == "llx" and seen["purpose"] == "parse"
    assert seen["parse"] == {"file_id": "f1", "tier": "agentic", "version": "latest", "expand": ["markdown"]}
    monkeypatch.setenv("PDF_BENCH_LLAMAPARSE_TIER", "fast")
    with pytest.raises(ValueError, match="no Markdown"):
        llamaparse.llamaparse_pages(REGISTRY["llamaparse"], pdf)


# --- Unstructured --------------------------------------------------------------------------------------------------

def el(category, text, page=1, **meta):
    return {"category": category, "text": text, "metadata": {"page_number": page, **meta}}


def test_unstructured_elements_to_pages():
    pages = unstructured.unstructured_elements_to_pages([
        el("Header", "ACME"), el("Title", "Report", category_depth=0), el("NarrativeText", "Intro."),
        el("Title", "Part", category_depth=1), el("ListItem", "one"), el("ListItem", "two"),
        el("Table", "A B 1 2", text_as_html="<table><tr><td>A</td></tr></table>"), el("Image", ""),
        el("FigureCaption", "Fig. 1"), el("Footer", "p1"), el("PageNumber", "1"),
        el("Title", "Next", page=2, category_depth=0), el("ListItem", "three", page=2)])
    assert len(pages) == 2 and "ACME" not in pages[0] and "p1" not in pages[0]
    assert kinds(pages[0]) == [("heading", 1), ("paragraph", None), ("heading", 2), ("list", None), ("table", None),
                               ("figure", None), ("paragraph", None)]
    assert "- one\n- two" in pages[0] and kinds(pages[1]) == [("heading", 1), ("list", None)]


def test_unstructured_objects_work_too():
    e = SimpleNamespace(category="Title", text="Hi", metadata=SimpleNamespace(page_number=1, category_depth=2))
    assert unstructured.unstructured_elements_to_pages([e]) == ["### Hi"]


# --- OpenDocRouter -------------------------------------------------------------------------------------------------

class FakeODR:
    """A stand-in for httpx.Client: records the calls, answers like the service."""

    def __init__(self, replies, log):
        self.replies, self.log = replies, log

    def __call__(self, base_url=None, headers=None, timeout=None):
        self.log["base_url"], self.log["auth"] = base_url, headers.get("Authorization")
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None):
        self.log.setdefault("calls", []).append((url, json))
        status, body, headers = self.replies.pop(0)
        return SimpleNamespace(status_code=status, headers=headers, json=lambda: body, text=str(body))


def ok_page(n, text):
    return {"page": n, "status": "ok", "markdown": text, "cached": False, "charge_usd": 0.002,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_opendocrouter_inline_document(pdf, monkeypatch):
    import httpx

    log = {}
    reply = {"status": "completed", "model_version": "v1", "charge_usd": 0.004,
             "pages": [ok_page(2, "# Two"), ok_page(1, "One")]}
    monkeypatch.setattr(httpx, "Client", FakeODR([(200, reply, {})], log))
    monkeypatch.setenv("OPEN_DOC_ROUTER_API_KEY", "odr-key")
    monkeypatch.setenv("PDF_BENCH_OPENDOCROUTER_MODEL", "google/gemini-3-flash")
    p = REGISTRY["opendocrouter"]
    assert p.hosted and not p.status()
    pages = opendocrouter.opendocrouter_pages(p, pdf)
    assert list(pages) == ["One", "# Two"]  # sorted by page number
    assert log["auth"] == "Bearer odr-key" and log["base_url"] == "https://www.opendocrouter.ai"
    url, body = log["calls"][0]
    assert url == "/v1/parse" and body["model"] == "google/gemini-3-flash" and "pages" not in body
    assert body["document"]["mime_type"] == "application/pdf" and body["document"]["data"]
    assert pages.raw["charge_usd"] == 0.004 and pages.raw["model_version"] == "v1"


def test_opendocrouter_failed_page_raises_and_errors_are_explained(pdf, monkeypatch):
    import httpx

    monkeypatch.setenv("PDF_BENCH_OPENDOCROUTER_MODEL", "m")
    p = REGISTRY["opendocrouter"]
    partial = {"charge_usd": 0, "pages": [ok_page(1, "x"), {"page": 2, "status": "error", "charge_usd": 0,
                                                            "error": {"code": "timeout", "message": "slow"}}]}
    monkeypatch.setattr(httpx, "Client", FakeODR([(200, partial, {})], {}))
    with pytest.raises(RuntimeError, match=r"1 page\(s\) failed .*p2 timeout"):
        opendocrouter.opendocrouter_pages(p, pdf)
    monkeypatch.setattr(httpx, "Client", FakeODR([(402, {"error": {"code": "insufficient_credits", "message": "top up"}}, {})], {}))
    with pytest.raises(RuntimeError, match="402 insufficient_credits: top up"):
        opendocrouter.opendocrouter_pages(p, pdf)


def test_opendocrouter_retries_when_busy_and_chunks_long_documents(tmp_path, monkeypatch):
    import httpx
    import pymupdf

    long = tmp_path / "long.pdf"
    d = pymupdf.open()
    for _ in range(5):
        d.new_page()
    d.save(long)
    log = {}
    replies = [(503, {"error": {"code": "at_capacity", "message": "busy"}}, {"Retry-After": "0"}),
               (200, {"charge_usd": 0.01, "pages": [ok_page(i, f"p{i}") for i in (1, 2, 3)]}, {}),
               (200, {"charge_usd": 0.01, "pages": [ok_page(i, f"p{i}") for i in (4, 5)]}, {})]
    monkeypatch.setattr(httpx, "Client", FakeODR(replies, log))
    monkeypatch.setenv("PDF_BENCH_OPENDOCROUTER_MODEL", "m")
    monkeypatch.setenv("PDF_BENCH_OPENDOCROUTER_PAGES_PER_REQUEST", "3")
    pages = opendocrouter.opendocrouter_pages(REGISTRY["opendocrouter"], long)
    assert list(pages) == ["p1", "p2", "p3", "p4", "p5"] and pages.raw["charge_usd"] == 0.02
    assert [c[1].get("pages") for c in log["calls"]] == ["1-3", "1-3", "4-5"]  # the first one was retried


def test_opendocrouter_big_file_goes_through_an_upload(pdf, monkeypatch):
    import httpx

    log = {}
    uploads = (201, {"upload_id": "u1", "upload_url": "https://up.example/x", "max_bytes": 1}, {})
    monkeypatch.setattr(httpx, "Client", FakeODR([uploads, (200, {"charge_usd": 0, "pages": [ok_page(1, "ok")]}, {})], log))
    monkeypatch.setattr(httpx, "put", lambda url, content=None, timeout=None: SimpleNamespace(raise_for_status=lambda: log.update(put=url)))
    monkeypatch.setattr(opendocrouter, "ODR_INLINE_MAX", 10)  # treat the test PDF as big
    monkeypatch.setenv("PDF_BENCH_OPENDOCROUTER_MODEL", "m")
    assert list(opendocrouter.opendocrouter_pages(REGISTRY["opendocrouter"], pdf)) == ["ok"]
    assert [c[0] for c in log["calls"]] == ["/v1/uploads", "/v1/parse"] and log["put"] == "https://up.example/x"
    assert log["calls"][1][1]["document"] == {"upload_id": "u1"}


def test_opendocrouter_model_list():
    payload = {"price_version": "x", "data": [
        {"id": "a/low", "name": "Low", "version": "1", "max_sync_pages": 50, "avg_charge_per_page_usd": 0.001,
         "parsebench": {"score": 60.0, "tables": 50.0, "charts": 1, "content_faithfulness": 2, "semantic_formatting": 3, "visual_grounding": 4}},
        {"id": "b/new", "name": "New", "version": "1", "max_sync_pages": 20, "avg_charge_per_page_usd": None, "parsebench": None},
        {"id": "c/high", "name": "High", "version": "2", "max_sync_pages": 50, "avg_charge_per_page_usd": 0.05,
         "parsebench": {"score": 84.2, "tables": 40.0, "charts": 1, "content_faithfulness": 2, "semantic_formatting": 3, "visual_grounding": 4}}]}
    assert [r["id"] for r in opendocrouter_util.models(payload)] == ["c/high", "a/low", "b/new"]  # unbenchmarked last
    assert [r["id"] for r in opendocrouter_util.models(payload, rank_by="tables")][:2] == ["a/low", "c/high"]
    top = opendocrouter_util.models(payload)[0]
    assert top["overall"] == 84.2 and top["usd_per_1k_pages"] == 50.0
    assert opendocrouter_util.label("google/gemini-3-flash") == "odr:gemini-3-flash"


# --- OpenDataLoader PDF ----------------------------------------------------------------------------------------

def test_java_major_reads_old_and_new_version_strings(monkeypatch):
    import subprocess
    import types

    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/java")
    for text, major in (('openjdk version "21.0.12.1" 2026-08-18', 21), ('java version "1.8.0_402"', 8),
                        ('openjdk version "11.0.2" 2019-01-15', 11), ("garbage", None)):
        monkeypatch.setattr(subprocess, "run", lambda *a, _t=text, **k: types.SimpleNamespace(stderr=_t, stdout=""))
        assert opendataloader.java_major() == major
    monkeypatch.setattr("shutil.which", lambda n: None)
    assert opendataloader.java_major() is None


def test_opendataloader_readiness_explains_java(monkeypatch):
    from pdf_bench.parsers import REGISTRY

    monkeypatch.setattr(opendataloader, "java_major", lambda: None)
    assert "Java" in REGISTRY["opendataloader"].status()
    monkeypatch.setattr(opendataloader, "java_major", lambda: 8)
    assert "Java 8" in REGISTRY["opendataloader"].status()


def test_opendataloader_adapter_with_fake_module(tmp_path, monkeypatch):
    import sys
    import types

    from pdf_bench.parsers import REGISTRY

    seen = {}

    def convert(input_path, output_dir, format, **kw):
        seen.update(kw, format=format)
        stem = Path(input_path[0]).stem
        if "html" in format.split(",")[0]:
            (Path(output_dir) / f"{stem}.html").write_text("<html><body>\n<h1>T</h1>\n</body></html>", encoding="utf-8")
        else:
            sep = kw["markdown_page_separator"].strip()
            (Path(output_dir) / f"{stem}.md").write_text(f"\n{sep.replace('%page-number%', '1')}\n# T\n{sep.replace('%page-number%', '2')}\nbody", encoding="utf-8")
        (Path(output_dir) / f"{stem}.json").write_text('{"kids": []}', encoding="utf-8")

    monkeypatch.setitem(sys.modules, "opendataloader_pdf", types.SimpleNamespace(convert=convert))
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"%PDF")
    pages = opendataloader.opendataloader_pages(REGISTRY["opendataloader"], pdf)
    assert pages == ["# T", "body"] and pages.raw == {"kids": []}  # default: Markdown with HTML tables, one text per page
    assert seen["format"] == "markdown,json" and seen["markdown_with_html"] is True and "hybrid" not in seen
    assert seen["image_output"] == "external"
    monkeypatch.setenv("PDF_BENCH_OPENDATALOADER_FORMAT", "html")
    assert opendataloader.opendataloader_pages(REGISTRY["opendataloader"], pdf) == ["<h1>T</h1>"]
    monkeypatch.delenv("PDF_BENCH_OPENDATALOADER_FORMAT")

    monkeypatch.setenv("PDF_BENCH_OPENDATALOADER_HYBRID", "docling-fast")
    monkeypatch.setenv("PDF_BENCH_OPENDATALOADER_HYBRID_URL", "http://localhost:5002")
    started = []
    monkeypatch.setattr(opendataloader, "ensure_hybrid_server", lambda url, **k: started.append(url))
    opendataloader.opendataloader_pages(REGISTRY["opendataloader"], pdf)
    assert seen["hybrid"] == "docling-fast" and seen["hybrid_url"] == "http://localhost:5002"
    assert started == ["http://localhost:5002"]
    monkeypatch.delenv("PDF_BENCH_OPENDATALOADER_HYBRID_URL")
    opendataloader.opendataloader_pages(REGISTRY["opendataloader"], pdf)
    assert seen["hybrid_url"] == opendataloader.HYBRID_URL  # default address when none is given


def test_hybrid_server_is_started_only_when_nothing_answers(monkeypatch):
    import pytest

    monkeypatch.setattr(opendataloader, "_port_open", lambda url: True)
    opendataloader.ensure_hybrid_server("http://somewhere.example:5002")  # already answering: nothing to start
    monkeypatch.setattr(opendataloader, "_port_open", lambda url: False)
    with pytest.raises(RuntimeError, match="no hybrid server answers"):
        opendataloader.ensure_hybrid_server("http://somewhere.example:5002")
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setattr("sys.executable", "/nonexistent/python")
    with pytest.raises(RuntimeError, match="not installed"):
        opendataloader.ensure_hybrid_server("http://localhost:5002")


def test_opendataloader_real_run_keeps_merged_cells(tmp_path):
    import importlib.util

    import pytest

    from pdf_bench import elements, score
    from pdf_bench.parsers import REGISTRY
    from pdf_bench.synth import generate_structured

    if importlib.util.find_spec("opendataloader_pdf") is None or (opendataloader.java_major() or 0) < 11:
        pytest.skip("needs opendataloader-pdf and Java 11+")
    pdf = generate_structured(tmp_path, only=["table_merged"])[0]
    pages = opendataloader.opendataloader_pages(REGISTRY["opendataloader"], pdf)
    report = score.score(elements.load_gt(pdf.with_suffix(".gt.json")), list(pages), parser="opendataloader")
    assert report.overall()["structure"] >= 0.9
