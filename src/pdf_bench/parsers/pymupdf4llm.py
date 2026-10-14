"""PyMuPDF4LLM, two settings: the defaults, and the maximum (HTML tables, header and footer dropped, OCR)."""

from __future__ import annotations

from pathlib import Path

from .base import entry


def pymupdf4llm_pages(parser, path: Path) -> list[str]:
    import pymupdf4llm

    # layout (default): better tables and headings, but the target of a link is lost; legacy keeps links
    pymupdf4llm.use_layout(parser.option("layout", "true").lower() != "false")
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    return [c["text"] for c in chunks]


def pymupdf4llm_max_pages(parser, path: Path) -> list[str]:
    """Layout model + HTML tables (keeps spans) + page header/footer dropped + automatic OCR."""
    import os

    import pymupdf4llm

    pymupdf4llm.use_layout(True)
    kwargs: dict = {"table_output": "html", "header": False, "footer": False}
    if os.environ.get("PDF_BENCH_OCR_LANG"):
        kwargs["ocr_language"] = os.environ["PDF_BENCH_OCR_LANG"]
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, **kwargs)
    return [c["text"] for c in chunks]


entry('pymupdf4llm', 'PyMuPDF4LLM (defaults)', "library", pymupdf4llm_pages,
      requires='pymupdf4llm',
      note='Default settings: layout model, Markdown tables (PyMuPDF is AGPL)',
      tested=True,
      version='1.28.2',
      license='AGPL-3.0 or commercial',
      source='https://github.com/pymupdf/pymupdf4llm',
      install='included',
      options={'layout': 'true'})

entry('pymupdf4llm_max', 'PyMuPDF4LLM (max settings)', "library", pymupdf4llm_max_pages,
      requires='pymupdf4llm',
      note='Layout model, HTML tables, header/footer dropped, auto OCR (PDF_BENCH_OCR_LANG)',
      tested=True,
      version='1.28.2',
      license='AGPL-3.0 or commercial',
      source='https://github.com/pymupdf/pymupdf4llm',
      install='included')
