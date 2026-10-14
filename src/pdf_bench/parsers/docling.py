"""Docling: layout model, TableFormer, OCR; heading levels and table models are settings."""

from __future__ import annotations

from pathlib import Path

from .base import entry


def docling_pages(parser, path: Path) -> list[str]:  # pragma: no cover - needs model weights from Hugging Face
    """Docling at full strength: layout model, TableFormer (accurate), cell matching, OCR."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    me = parser
    model, matching = me.option("table_model", "tableformer").lower(), me.option("cell_matching", "true").lower() != "false"
    if model == "tableformer":  # V1, accurate mode
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        opts.table_structure_options.do_cell_matching = matching
    else:
        from docling.datamodel import pipeline_options as po

        if model == "tableformer_v2":
            opts.table_structure_options = po.TableStructureV2Options(do_cell_matching=matching)
        elif model == "granite_vision":  # a vision-language model reads the table (downloads a large model)
            opts.table_structure_options = po.GraniteVisionTableStructureOptions()
        else:
            raise ValueError(f"unknown table_model {model!r}: tableformer, tableformer_v2 or granite_vision")
    if parser.option("heading_hierarchy", "true").lower() != "false":
        # off by default in Docling, which then gives every heading level 1: levels from bookmarks, numbering, font style
        try:
            from docling.datamodel.pipeline_options import HeadingHierarchyOptions
        except ImportError as exc:
            raise RuntimeError("this Docling has no HeadingHierarchyOptions: upgrade docling, or set "
                               "PDF_BENCH_DOCLING_HEADING_HIERARCHY=false") from exc
        opts.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
        opts.generate_parsed_pages = True  # the font-style signal needs the parsed pages
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = conv.convert(str(path)).document
    if parser.option("format", "markdown") == "html":
        # Markdown cannot express merged cells (Docling writes "Plan - Monthly" in each column); its HTML export keeps
        # rowspan/colspan, headings, lists and figures as tags
        from .html_util import html_body

        return [html_body(doc.export_to_html(page_no=n)) for n in sorted(doc.pages)]
    return [doc.export_to_markdown(page_no=n) for n in sorted(doc.pages)]


entry('docling', 'Docling', "library", docling_pages,
      requires='docling',
      note="TableFormer accurate + OCR. Needs HF model weights; not run in this repo's sandbox. Setting format = markdown (default) or html: HTML keeps merged cells (rowspan/colspan), Markdown cannot. Setting heading_hierarchy = true (default): heading levels from PDF bookmarks, numbering and font style (Docling leaves every heading at level 1 otherwise). Settings table_model = tableformer (default) | tableformer_v2 | granite_vision; cell_matching = true | false (false lets the table model define the cells instead of snapping them to the PDF's text cells: try it when merged cells break)",
      tested=False,
      version='2.135.0',
      license='MIT',
      source='https://github.com/docling-project/docling',
      install='included',
      reference='PDF-Bench: 71.3% edit similarity',
      options={'format': 'markdown', 'heading_hierarchy': 'true', 'table_model': 'tableformer', 'cell_matching': 'true'})
