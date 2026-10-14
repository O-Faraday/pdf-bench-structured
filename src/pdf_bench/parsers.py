"""Registry of the structured-extraction parsers.

Each parser turns a PDF into one string per page (Markdown or HTML) and is timed. Only parsers that
recover document structure (headings, tables, lists...) are listed; plain text extractors are out of
scope. A parser whose library, tool or credentials are missing reports why instead of crashing a run.
The candidates beyond the two local libraries below are declared in :mod:`pdf_bench.catalog`.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


def _importable(module: str) -> bool:
    try:  # a dotted name imports its parents, which raises when the top package is missing
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class ParseResult:
    parser: str
    path: str
    pages: list[str]
    seconds: float
    error: str | None = None
    raw: object = None  # the service's raw response, when the adapter keeps it

    @property
    def text(self) -> str:
        return "\n\n".join(self.pages)

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Parser:
    name: str
    fn: Callable[[Path], list[str]]
    output: str  # "markdown" | "text"
    requires: str  # importable module name ("" when the tool is a command line program)
    note: str = ""
    tested: bool = True
    hosted: bool = False  # sends documents to a remote service
    # --- catalogue metadata (see pdf_bench.catalog) ---
    label: str = ""
    family: str = "library"  # library | hosted_vlm | cloud_service
    version: str = ""  # latest version found when the catalogue was last checked
    install: str = ""  # how to install it
    needs: tuple[str, ...] = ()  # environment variables that must be set (credentials, endpoints)
    gpu: bool = False
    license: str = ""
    source: str = ""
    reference: str = ""  # published score, for orientation only
    options: dict = field(default_factory=dict)  # defaults such as the model id
    check: Callable[[], str] | None = None  # extra readiness test: "" when ready, else the reason
    implemented: bool = True
    # element kinds this tool's output cannot express at all (kind -> why, from its documentation): their structure
    # score is "not applicable" instead of 0. Never set from observed results, which would be circular.
    cannot: dict = field(default_factory=dict)

    def option(self, key: str, default: str = "") -> str:
        """Setting ``key``: env ``PDF_BENCH_<PARSER>_<KEY>`` first, then the catalogue default."""
        from . import config

        env = f"PDF_BENCH_{self.name}_{key}".upper().replace("-", "_")
        return config.get(env) or str(self.options.get(key, default))

    def status(self) -> str:
        """Empty when the parser can run here, otherwise why it cannot (shown in the app)."""
        from . import config

        if not self.implemented:
            return "adapter not implemented yet"
        if self.requires and not _importable(self.requires):
            return f"not installed: {self.install}" if self.install else "not installed"
        if self.check is not None and (reason := self.check()):
            return reason
        missing = [v for v in self.needs if not config.get(v)]
        if missing:
            return "missing in .env: " + ", ".join(missing)
        return ""

    def available(self) -> bool:
        return self.status() == ""

    def parse(self, path: str | Path) -> ParseResult:
        path = Path(path)
        start = time.perf_counter()
        try:
            pages = self.fn(path)
            err, raw = None, getattr(pages, "raw", None)
            pages = list(pages)
        except Exception as exc:  # noqa: BLE001 - a parser failure is a result, not a crash
            pages, err, raw = [], f"{type(exc).__name__}: {exc}", None
        return ParseResult(self.name, str(path), pages, time.perf_counter() - start, err, raw)


# --- adapters ---------------------------------------------------------------

def _pymupdf4llm(path: Path) -> list[str]:
    import pymupdf4llm

    # layout (default): better tables and headings, but the target of a link is lost; legacy keeps links
    pymupdf4llm.use_layout(REGISTRY["pymupdf4llm"].option("layout", "true").lower() != "false")
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [c["text"] for c in chunks]


def _pymupdf4llm_max(path: Path) -> list[str]:
    """Layout model + HTML tables (keeps spans) + page header/footer dropped + automatic OCR."""
    import os

    import pymupdf4llm

    pymupdf4llm.use_layout(True)
    kwargs: dict = {"table_output": "html", "header": False, "footer": False}
    if os.environ.get("PDF_BENCH_OCR_LANG"):
        kwargs["ocr_language"] = os.environ["PDF_BENCH_OCR_LANG"]
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, **kwargs)
    return [c["text"] for c in chunks]


def _docling(path: Path) -> list[str]:  # pragma: no cover - needs model weights from Hugging Face
    """Docling at full strength: layout model, TableFormer (accurate), cell matching, OCR."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = conv.convert(str(path)).document
    if REGISTRY["docling"].option("format", "markdown") == "html":
        # Markdown cannot express merged cells (Docling writes "Plan - Monthly" in each column); its HTML export keeps
        # rowspan/colspan, headings, lists and figures as tags
        from .html_util import html_body

        return [html_body(doc.export_to_html(page_no=n)) for n in sorted(doc.pages)]
    return [doc.export_to_markdown(page_no=n) for n in sorted(doc.pages)]


REGISTRY: dict[str, Parser] = {}
# The oracles read the ground truth: they are score ceilings, not parsers, so they live apart from REGISTRY.
REFERENCES: dict[str, Parser] = {}


def lookup(name: str) -> Parser:
    """A parser or a reference by name."""
    return REGISTRY[name] if name in REGISTRY else REFERENCES[name]


def register(parser: Parser) -> Parser:
    REGISTRY[parser.name] = parser
    return parser


for _p in (
    Parser("pymupdf4llm", _pymupdf4llm, "markdown", "pymupdf4llm",
           "Default settings: layout model, Markdown tables (PyMuPDF is AGPL)",
           label="PyMuPDF4LLM (defaults)", version="1.28.2", license="AGPL-3.0 or commercial", options={"layout": "true"},
           source="https://github.com/pymupdf/pymupdf4llm", install="included"),
    Parser("pymupdf4llm_max", _pymupdf4llm_max, "markdown", "pymupdf4llm",
           "Layout model, HTML tables, header/footer dropped, auto OCR (PDF_BENCH_OCR_LANG)",
           label="PyMuPDF4LLM (max settings)", version="1.28.2", license="AGPL-3.0 or commercial",
           source="https://github.com/pymupdf/pymupdf4llm", install="included"),
    Parser("docling", _docling, "markdown", "docling",
           "TableFormer accurate + OCR. Needs HF model weights; not run in this repo's sandbox. Setting format = markdown (default) or html: "
           "HTML keeps merged cells (rowspan/colspan), Markdown cannot",
           tested=False, label="Docling", version="2.135.0", license="MIT",
           source="https://github.com/docling-project/docling",
           install="included", reference="PDF-Bench: 71.3% edit similarity", options={"format": "markdown"}),
):
    register(_p)

# the catalogue adds the other candidates (libraries, local VLMs, hosted models, cloud services)
from . import catalog as _catalog  # noqa: E402,F401


def available_parsers(names: list[str] | None = None) -> list[Parser]:
    """Parsers that can run. Without ``names``, hosted ones are left out: they send the PDFs to a
    third party and cost money, so they run only when asked for by name."""
    chosen = [REGISTRY[n] for n in names] if names is not None else [p for p in REGISTRY.values() if not p.hosted]
    return [p for p in chosen if p.available()]


def unavailable(names: list[str] | None = None) -> dict[str, str]:
    """Requested (or all) parsers that cannot run here, with the reason."""
    chosen = [REGISTRY[n] for n in names] if names is not None else list(REGISTRY.values())
    return {p.name: p.status() for p in chosen if p.status()}


def unknown(names: list[str]) -> list[str]:
    return [n for n in names if n not in REGISTRY]


def run_parsers(path: str | Path, names: list[str] | None = None) -> list[ParseResult]:
    return [p.parse(path) for p in available_parsers(names)]
