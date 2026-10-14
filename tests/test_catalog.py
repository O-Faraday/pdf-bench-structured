"""Catalogue and adapters. Only fake clients here: no network, GPU or credentials."""

import sys
import types
from pathlib import Path

import pytest

from pdf_bench import adapters
from pdf_bench.parsers import REGISTRY, available_parsers, unavailable, unknown
from pdf_bench.synth import generate_structured


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    d = tmp_path_factory.mktemp("structured")
    return generate_structured(d, only=["table_simple"])[0]


def test_only_structured_parsers_are_listed():
    plain = {"pypdf", "pdfplumber", "pdfplumber_max", "pdfium", "pymupdf", "pymupdf_max", "markitdown"}
    assert not plain & set(REGISTRY)
    assert {"pymupdf4llm", "docling", "databricks_ai_parse", "landingai_ade"} <= set(REGISTRY)
    for p in REGISTRY.values():
        assert p.label and p.family in {"library", "hosted_vlm", "cloud_service"}, p.name


def test_hosted_flags_and_default_set():
    assert all(REGISTRY[n].hosted for n in ("claude", "gemini", "gpt", "mistral_ocr", "databricks_ai_parse", "llamaparse", "opendocrouter",
                                            "landingai_ade"))
    assert not any(p.hosted for p in available_parsers())
    assert unavailable(["aws_textract"])["aws_textract"].startswith("adapter not implemented")
    assert unknown(["nope", "claude"]) == ["nope"]


def test_missing_credentials_are_explained(monkeypatch):
    monkeypatch.setattr("pdf_bench.config.get", lambda n, d=None: None)
    assert "VISION_AGENT_API_KEY" in REGISTRY["landingai_ade"].status()
    assert "ANTHROPIC_API_KEY" in REGISTRY["claude"].status()
    assert "VOLUME" in REGISTRY["databricks_ai_parse"].status()


def test_option_override_from_environment(monkeypatch):
    monkeypatch.setenv("PDF_BENCH_CLAUDE_MODEL", "other-model")
    assert REGISTRY["claude"].option("model") == "other-model"


def test_strip_fence():
    assert adapters.strip_fence("```markdown\n# A\n```") == "# A"
    assert adapters.strip_fence("plain") == "plain"


def test_page_adapters_with_fake_clients(pdf, monkeypatch):
    calls = {}

    class FakeAnthropic:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                calls["anthropic"] = kw
                block = types.SimpleNamespace(type="text", text="```markdown\n# Title\n```")
                return types.SimpleNamespace(content=[block])

    class FakeOpenAI:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    calls["openai"] = kw
                    msg = types.SimpleNamespace(content="| a |\n|---|\n| 1 |")
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setattr(adapters, "_anthropic_client", lambda: FakeAnthropic)
    monkeypatch.setattr(adapters, "_openai_client", lambda base_url=None, api_key=None: FakeOpenAI)
    assert adapters.anthropic_pages(REGISTRY["claude"], pdf) == ["# Title"]
    assert calls["anthropic"]["messages"][0]["content"][0]["type"] == "image"
    adapters.openai_compat_pages(REGISTRY["gpt"], pdf)
    assert "max_completion_tokens" in calls["openai"]


def test_databricks_elements_to_markdown():
    parsed = {"document": {
        "pages": [{"id": 0}, {"id": 1}],
        "elements": [
            {"id": 0, "type": "page_header", "content": "ACME", "bbox": [{"page_id": 0}]},
            {"id": 1, "type": "title", "content": "Report", "bbox": [{"page_id": 0}]},
            {"id": 2, "type": "section_header", "content": "Scope", "bbox": [{"page_id": 0}]},
            {"id": 3, "type": "text", "content": "Body text.", "bbox": [{"page_id": 0}]},
            {"id": 4, "type": "table", "content": "<table><tr><td>1</td></tr></table>", "bbox": [{"page_id": 1}]},
            {"id": 5, "type": "figure", "content": None, "bbox": [{"page_id": 1}]},
            {"id": 6, "type": "page_number", "content": "2", "bbox": [{"page_id": 1}]},
        ]}}
    p0, p1 = adapters.databricks_elements_to_pages(parsed)
    assert p0 == "# Report\n\n## Scope\n\nBody text."
    assert p1 == "<table><tr><td>1</td></tr></table>\n\n[image]"


def test_databricks_adapter_with_fake_workspace(pdf, monkeypatch):
    import json

    seen = {}

    class Files:
        def upload(self, path, fh, overwrite=None):
            seen["upload"] = (path, fh.read(4))

        def delete(self, path):
            seen["deleted"] = path

    class Stmt:
        def execute_statement(self, statement, warehouse_id, wait_timeout=None):
            seen["sql"], seen["wh"] = statement, warehouse_id
            payload = json.dumps({"document": {"pages": [{"id": 0}], "elements": [
                {"id": 0, "type": "text", "content": "Hello", "bbox": [{"page_id": 0}]}]}})
            state = types.SimpleNamespace(value="SUCCEEDED")
            return types.SimpleNamespace(status=types.SimpleNamespace(state=state), statement_id="s",
                                         result=types.SimpleNamespace(data_array=[[payload]]))

    class FakeWorkspace:
        files, statement_execution = Files(), Stmt()

    monkeypatch.setattr("databricks.sdk.WorkspaceClient", FakeWorkspace)
    monkeypatch.setenv("PDF_BENCH_DATABRICKS_AI_PARSE_VOLUME", "/Volumes/main/default/bench")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "wh123")
    assert adapters.databricks_pages(REGISTRY["databricks_ai_parse"], pdf) == ["Hello"]
    assert seen["upload"][0].startswith("/Volumes/main/default/bench/") and seen["upload"][1] == b"%PDF"
    assert "ai_parse_document" in seen["sql"] and "'2.0'" in seen["sql"] and seen["wh"] == "wh123"
    assert seen["deleted"] == seen["upload"][0]


def test_landingai_chunks_to_markdown_and_adapter(pdf, monkeypatch):
    def chunk(kind, md, page):
        return types.SimpleNamespace(type=kind, markdown=md, grounding=types.SimpleNamespace(page=page))

    chunks = [chunk("marginalia", "page 1", 0), chunk("text", "<a id='x1'></a>\n\n# Title", 0),
              chunk("figure", "<::a chart::>", 1), chunk("table", "<a id='t'></a><table></table>", 1)]
    assert adapters.landingai_chunks_to_pages(chunks) == ["# Title", "[image]\n\na chart\n\n<table></table>"]

    class FakeClient:
        def __init__(self, **kw):
            pass

        def parse(self, document, model):
            assert model == "dpt-2-latest" and Path(document).exists()
            return types.SimpleNamespace(chunks=chunks, markdown="whole")

    monkeypatch.setattr("landingai_ade.LandingAIADE", FakeClient)
    assert adapters.landingai_pages(REGISTRY["landingai_ade"], pdf)[0] == "# Title"


def test_empty_request_means_nothing_not_everything():
    assert unavailable([]) == {} and available_parsers([]) == []



def test_openai_key_is_not_sent_to_other_servers(monkeypatch):
    import openai
    from pdf_bench import adapters

    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert adapters._openai_client("http://localhost:8000/v1").api_key == "EMPTY"
    assert adapters._openai_client(None).api_key == "sk-openai"


def test_landingai_keeps_footnotes_captions_and_signatures():
    def chunk(kind, md, page):
        return types.SimpleNamespace(type=kind, markdown=md, grounding=types.SimpleNamespace(page=page))

    chunks = [chunk("marginalia", "ACME - File 2025", 0), chunk("marginalia", "Page 1", 0),
              chunk("marginalia", "1 The deductible is set in the contract.", 0),
              chunk("figure", "<a id='f'></a><::Figure 1: Claims per quarter. A bar chart.: figure::>", 0),
              chunk("attestation", "<::attestation: Signature block\nStatus: unsigned\nReadable Text: Signatures, Driver A signature::>", 0),
              chunk("marginalia", "ACME - File 2025", 1), chunk("marginalia", "Page 2", 1), chunk("text", "Body", 1)]
    pages = adapters.landingai_chunks_to_pages(chunks)
    assert "1 The deductible is set in the contract." in pages[0]            # a footnote is margin text too
    assert "ACME - File 2025" not in "".join(pages) and "Page 1" not in pages[0]  # running header and page numbers go
    assert "Figure 1: Claims per quarter" in pages[0] and "[image]" in pages[0]
    assert "Signatures, Driver A signature" in pages[0] and "[signature]" in pages[0]
