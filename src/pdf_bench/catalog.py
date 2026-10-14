"""Catalogue of the structured-extraction parsers that can take part in a benchmark.

Only tools that recover document structure (headings, tables, lists, figures...) are listed; plain
text extractors are out of scope. Every entry becomes a :class:`pdf_bench.parsers.Parser` in
``REGISTRY`` and carries: family, version, how to install it, what it needs (GPU, credentials),
licence, source and a published score.

Versions of Python packages were read from PyPI on ``CHECKED``. Hosted-model ids and Hugging Face
repositories change often and could not all be confirmed from a primary source: the id is a
default, editable with ``PDF_BENCH_<PARSER>_MODEL`` or the model's settings in the app. Entries without ``tested`` have
never been run against the real tool or service here.
"""

from __future__ import annotations

import functools

from . import adapters as ad
from . import oracle
from .parsers import REFERENCES, REGISTRY, Parser, register

CHECKED = "2026-10-07"

FAMILIES = {
    "library": "Libraries and pipelines (local)",
    "hosted_vlm": "Hosted models (documents leave your machine, paid)",
    "cloud_service": "Cloud document services",
}

def _get(name: str) -> Parser:
    return REGISTRY[name]


def _entry(name: str, label: str, family: str, adapter, **kw) -> Parser:
    """Build and register a catalogue parser. ``adapter`` is called as adapter(parser, path)."""
    implemented = adapter is not None
    holder: dict[str, Parser] = {}

    def fn(path):
        if not implemented:
            raise NotImplementedError("adapter not implemented yet")
        return adapter(holder["p"], path)

    check = kw.pop("check", None)
    p = Parser(name, fn, kw.pop("output", "markdown"), kw.pop("requires", ""), kw.pop("note", ""),
               tested=kw.pop("tested", False), label=label, family=family, implemented=implemented,
               **kw)
    holder["p"] = p
    if check is not None:
        p.check = check(functools.partial(_get, name))
    return register(p)


def _reference(name: str, *args, **kw) -> Parser:
    """A reference (oracle): built like a parser, kept out of the parser catalogue."""
    p = _entry(name, *args, **kw)
    REFERENCES[name] = REGISTRY.pop(name)
    return p


def _any_env(*names: str):
    """Readiness: at least one of the environment variables is set."""

    def factory(_getter):
        def check() -> str:
            from . import config

            return "" if any(config.get(n) for n in names) else "missing in .env: " + " or ".join(names)

        return check

    return factory


def _chain(*checks):
    def check() -> str:
        for c in checks:
            if reason := c():
                return reason
        return ""

    return check


def _model_required(_getter):
    def check() -> str:
        p = _getter()
        return "" if p.option("model") else f"set PDF_BENCH_{p.name.upper()}_MODEL (or the model setting in the app) to the model id"

    return check


# --- hosted models ------------------------------------------------------------------------------------

_entry("claude", "Claude (Anthropic)", "hosted_vlm", ad.anthropic_pages, requires="anthropic",
       needs=("ANTHROPIC_API_KEY",), hosted=True, options={"model": "claude-sonnet-5-5", "dpi": 150},
       cannot={"link": "the model reads page images: a link target is not visible in the picture"},
       install="included", version="claude-sonnet-5-5", license="commercial API",
       source="https://docs.claude.com", reference="Claude Sonnet 4.6: in arXiv 2603.18652 (tables)",
       note="Page images sent to the Anthropic API. Other ids: claude-opus-5-5, claude-fable-5-1")

_entry("gemini", "Gemini (Google)", "hosted_vlm", ad.gemini_pages, requires="google.genai",
       hosted=True, options={"model": "gemini-3.8-flash", "dpi": 150},
       cannot={"link": "the model reads page images: a link target is not visible in the picture"},
       install="uv add google-genai", version="gemini-3.8-flash (id from a third-party list: verify)",
       license="commercial API", source="https://ai.google.dev/gemini-api/docs/models",
       reference="OmniDocBench v1.6: 92.91 (Gemini 3 Pro), 92.62 (Gemini 3 Flash)",
       check=_any_env("GEMINI_API_KEY", "GOOGLE_API_KEY"),
       note="Also seen: gemini-3.1-pro-preview. Verify the id in Google's documentation")

_entry("gpt", "GPT (OpenAI)", "hosted_vlm", ad.openai_compat_pages, requires="openai",
       needs=("OPENAI_API_KEY",), hosted=True, options={"model": "gpt-6-astra", "dpi": 150},
       cannot={"link": "the model reads page images: a link target is not visible in the picture"},
       install="uv add openai", version="gpt-6-astra (id from OpenAI's model list: verify image input)",
       license="commercial API", source="https://developers.openai.com/api/docs/models/all",
       reference="OmniDocBench v1.6: 86.59 (GPT-5.2)",
       note="Same list: gpt-6.1-sol (cheaper), gpt-6-luna (cheapest). Image input was not confirmed on "
            "the page read")

_entry("opendocrouter", "OpenDocRouter (one API, many parsing models)", "hosted_vlm", ad.opendocrouter_pages,
       requires="httpx", needs=("OPEN_DOC_ROUTER_API_KEY",), hosted=True,
       options={"model": "", "pages_per_request": 50}, check=lambda g: _model_required(g),
       install="included", version="the model id you give", license="depends on the model",
       source="https://www.opendocrouter.ai/",
       note="Made by LlamaIndex. Serves document-parsing specialists (MinerU2.5-Pro, PaddleOCR-VL, LightOnOCR, "
            "dots.mocr...) and frontier models behind one parse endpoint; only successful pages are billed. "
            "The app lists the models with their ParseBench scores (published by the service itself): make one "
            "named variant per model id (Models tab). Documents leave your machine. 'pages_per_request' must not "
            "exceed the model's max_sync_pages")

_entry("mistral_ocr", "Mistral OCR", "hosted_vlm", ad.mistral_ocr_pages, requires="mistralai",
       needs=("MISTRAL_API_KEY",), hosted=True, options={"model": "mistral-ocr-latest"},
       install="uv add mistralai", version="mistral-ocr-latest (alias)", license="commercial API",
       source="https://docs.mistral.ai", reference="OmniDocBench v1.6: 85.66; olmOCR-Bench: 72.0",
       note="The alias always points to the newest Mistral OCR model; the response is split by page")

# --- cloud document services ---------------------------------------------------------------------------------

def _databricks_ready(getter):
    def check() -> str:
        from . import config

        p = getter()
        if not p.option("volume"):
            return ("no volume chosen: pick one in the app or set PDF_BENCH_DATABRICKS_AI_PARSE_VOLUME "
                    "(e.g. /Volumes/main/default/pdf_bench)")
        if not (p.option("warehouse_id") or config.get("DATABRICKS_WAREHOUSE_ID")):
            return "no SQL warehouse chosen: pick one in the app or set DATABRICKS_WAREHOUSE_ID"
        return ""

    return check


_entry("databricks_ai_parse", "Databricks ai_parse_document", "cloud_service", ad.databricks_pages,
       requires="databricks.sdk", hosted=True, install="uv add databricks-sdk",
       needs=("DATABRICKS_HOST", "DATABRICKS_TOKEN"),
       options={"volume": "", "warehouse_id": "", "version": "2.0"}, version="output schema 2.0, adapter 2",
       license="commercial (Databricks)",
       source="https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document",
       reference="none found from a neutral source", check=_databricks_ready,
       cannot={"list": "the ai_parse_document schema has no list element: list items arrive as text",
               "link": "the schema has no hyperlink element"},
       note="SQL function: the PDF is uploaded to a Unity Catalog volume and parsed on a SQL warehouse "
            "(Runtime 17.3+ or serverless, region-limited, 500 pages and 100 MB per file, billed under "
            "AI_FUNCTIONS). Element types: text, table (HTML), figure, title, section_header, caption, "
            "footnote, page_header, page_footer, page_number, signature. Heading levels are not provided")

_entry("landingai_ade", "LandingAI Agentic Document Extraction", "cloud_service", ad.landingai_pages,
       requires="landingai_ade", hosted=True, needs=("VISION_AGENT_API_KEY",), install="uv add landingai-ade",
       options={"model": "dpt-2-latest", "region": "us"}, version="landingai-ade 1.18.1, model dpt-2-latest, adapter 2",
       license="commercial (LandingAI)", source="https://docs.landing.ai/ade/ade-python",
       reference="none found from a neutral source",
       note="Pinned snapshots exist (dpt-2-20260410, dpt-2-20260903...): pin one for reproducible runs. "
            "PDF_BENCH_LANDINGAI_ADE_REGION=eu for an EU key. Marginalia chunks (headers, footers) are dropped "
            "are kept unless they are page numbers or repeat on several pages (a footnote is margin text too). A figure is "
            "written as [image] followed by its description/caption; a signature block gives its readable text and [signature]. "
            "Heading levels are not provided (all headings come as '#')")

# --- more services: LlamaParse, Unstructured ------------------------------------------------------------------

_entry("llamaparse", "LlamaParse (LlamaCloud)", "cloud_service", ad.llamaparse_pages, requires="llama_cloud",
       hosted=True, needs=("LLAMA_CLOUD_API_KEY",), install='uv add "llama-cloud>=2.8"',
       options={"tier": "agentic", "version": "latest"}, version="llama-cloud >= 2.8, parse API v2",
       license="commercial (LlamaIndex)", source="https://developers.llamaindex.ai/llamaparse/parse/getting_started/",
       note="Tiers: cost_effective, agentic (default), agentic_plus; 'fast' returns no Markdown. 'version' latest "
            "changes over time: pin a version for reproducible runs. Billed per page, more for higher tiers")

_entry("unstructured", "Unstructured (partition_pdf)", "library", ad.unstructured_pages, requires="unstructured",
       install='uv add "unstructured[pdf]"   (hi_res also needs poppler and tesseract on the PATH)',
       options={"strategy": "hi_res"}, version="latest release (verify)", license="Apache-2.0",
       source="https://github.com/Unstructured-IO/unstructured",
       note="Heavy install. strategy: hi_res (layout model, tables), fast (text layer only) or ocr_only. Titles "
            "become headings by category_depth, tables come as HTML, there is no footnote element. On Windows, "
            "poppler and tesseract are the usual stumbling block: WSL is easier")

def _java_ready(_getter):
    def check() -> str:
        major = ad.java_major()
        if major is None:
            return "Java not found on the PATH: install Java 11 or later (e.g. Temurin) and reopen the terminal"
        if major < 11:
            return f"Java {major} found: OpenDataLoader needs Java 11 or later"
        return ""

    return check


_entry("opendataloader", "OpenDataLoader PDF", "library", ad.opendataloader_pages, requires="opendataloader_pdf",
       install='uv add "opendataloader-pdf[hybrid]"   (needs Java 11+ on the PATH)', check=_java_ready,
       options={"format": "markdown_html", "hybrid": "", "hybrid_url": "", "hybrid_mode": "", "use_struct_tree": "",
                "include_header_footer": ""},
       version="opendataloader-pdf 2.5.12, adapter 3", license="Apache-2.0", tested=True,
       cannot={"footnote": "no footnote element among its documented element types (heading, paragraph, table, list, "
                           "image, caption, formula)",
               "link": "hyperlinks are not among its documented element types"},
       source="https://github.com/opendataloader-project/opendataloader-pdf",
       reference="own benchmark (opendataloader-bench, 200 PDFs, run by the publisher): hybrid 0.907 overall",
       note="Java engine, no GPU, local mode by default. Output is Markdown with HTML tables (keeps rowspan/colspan, split by page); 'format' html is one text for the "
            "whole document, markdown loses merged cells. Links, footnote markers and form fields are not in its output. Hybrid mode "
            "(setting hybrid=docling-fast, needs \"opendataloader-pdf[hybrid]\", already a dependency of this project): "
            "the app starts the local hybrid server by itself on port 5002 (or uses hybrid_url if one answers there; "
            "the first start loads the Docling models, which can take minutes). There is no silent fallback to local "
            "mode: a failed page is an error. Each call starts a JVM: slow "
            "for many tiny files. The output is one text for the whole document, not split by page")

# --- references: the ceiling of each output format ----------------------------------------------------------------

_reference("oracle_html", "Oracle (ground truth as HTML)", "reference", oracle.oracle_html_pages, tested=True,
       install="included", version="-", license="-",
       note="Writes the ground truth as HTML. Scores 1.0 by construction: it proves the scorer, and shows what "
            "a perfect HTML answer looks like")
_reference("oracle_markdown", "Oracle (ground truth as Markdown)", "reference", oracle.oracle_markdown_pages, tested=True,
       install="included", version="-", license="-",
       note="Writes the ground truth as GitHub Markdown: the ceiling of Markdown, which cannot express merged "
            "or nested cells")

# --- listed, not implemented -------------------------------------------------------------------------

for _name, _label, _fam, _why, _src in (
    ("aws_textract", "AWS Textract", "cloud_service", "cloud service, needs AWS credentials",
     "https://aws.amazon.com/textract/"),
    ("google_docai", "Google Document AI", "cloud_service", "cloud service, needs a Google Cloud project",
     "https://cloud.google.com/document-ai"),
    ("mathpix", "Mathpix", "cloud_service", "hosted API", "https://mathpix.com"),
):
    _entry(_name, _label, _fam, None, source=_src, note=f"Not implemented: {_why}")
