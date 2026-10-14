"""Databricks ``ai_parse_document``: the PDF goes to a Unity Catalog volume, a SQL warehouse parses it."""

from __future__ import annotations

import re
from pathlib import Path

from .. import config
from .base import entry
from .common import Pages, int_option


def databricks_elements_to_pages(parsed: dict) -> list[str]:
    """Markdown per page from the ``ai_parse_document`` output (schema 2.0).

    Section headers and titles become headings (the service gives no level, so titles are ``#`` and
    section headers ``##``), tables stay HTML, figures become ``[image]`` (generated descriptions are
    not requested), and running page headers, footers and page numbers are left out.
    """
    doc = parsed.get("document", parsed)
    n_pages = max([p.get("id", 0) for p in doc.get("pages", [])] + [0]) + 1
    pages: list[list[str]] = [[] for _ in range(n_pages)]
    notes = 0
    for el in doc.get("elements", []):
        kind, text = el.get("type", ""), (el.get("content") or "").strip()
        bbox = el.get("bbox") or []
        page = int(bbox[0].get("page_id", 0)) if bbox else 0
        while page >= len(pages):
            pages.append([])
        if kind in ("page_header", "page_footer", "page_number"):
            continue
        if kind == "title":
            out = f"# {text}"
        elif kind == "section_header":
            out = f"## {text}"
        elif kind == "figure":
            out = "[image]"
        elif kind == "signature":
            out = "[signature]"
        elif kind == "footnote":  # one element can hold several notes, one per line ("1 ...", "2 ...")
            lines = []
            for line in (x.strip() for x in text.splitlines() if x.strip()):
                notes += 1
                m = re.match(r"^[\[\(]?(\d+|[*†‡])[\]\).]?\s+(.+)$", line)
                lines.append(f"[^{m.group(1) if m else notes}]: {m.group(2) if m else line}")
            out = "\n\n".join(lines)
        else:  # text, table (HTML), caption, footnote
            out = text
        if out:
            pages[page].append(out)
    return ["\n\n".join(p) for p in pages]


def databricks_pages(parser, path: Path) -> list[str]:
    """Upload the PDF to a Unity Catalog volume, run ``ai_parse_document`` on a SQL warehouse."""
    import json
    import time
    import uuid

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    volume = parser.option("volume").rstrip("/")
    remote = f"{volume}/{uuid.uuid4().hex}.pdf"
    with open(path, "rb") as fh:
        w.files.upload(remote, fh, overwrite=True)
    try:
        version = parser.option("version", "2.0")
        sql = (f"SELECT CAST(ai_parse_document(content, map('version', '{version}', "
               f"'descriptionElementTypes', '')) AS STRING) AS parsed "
               f"FROM READ_FILES('{remote.replace(chr(39), '')}', format => 'binaryFile')")
        resp = w.statement_execution.execute_statement(
            statement=sql, warehouse_id=parser.option("warehouse_id") or config.get("DATABRICKS_WAREHOUSE_ID"),
            wait_timeout="50s")
        deadline = time.time() + int_option(parser, "timeout", 900)
        while resp.status.state.value in ("PENDING", "RUNNING") and time.time() < deadline:
            time.sleep(3)
            resp = w.statement_execution.get_statement(resp.statement_id)
        state = resp.status.state.value
        if state != "SUCCEEDED":
            err = getattr(resp.status, "error", None)
            raise RuntimeError(f"ai_parse_document {state}: {getattr(err, 'message', '')}")
        parsed = json.loads(resp.result.data_array[0][0])
        pages = Pages(databricks_elements_to_pages(parsed))
        pages.raw = parsed
        return pages
    finally:
        try:
            w.files.delete(remote)
        except Exception:  # noqa: BLE001 - cleanup only
            pass


def _databricks_ready(getter):
    def check() -> str:
        p = getter()
        if not p.option("volume"):
            return ("no volume chosen: pick one in the app or set PDF_BENCH_DATABRICKS_AI_PARSE_VOLUME "
                    "(e.g. /Volumes/main/default/pdf_bench)")
        if not (p.option("warehouse_id") or config.get("DATABRICKS_WAREHOUSE_ID")):
            return "no SQL warehouse chosen: pick one in the app or set DATABRICKS_WAREHOUSE_ID"
        return ""

    return check


entry("databricks_ai_parse", "Databricks ai_parse_document", "cloud_service", databricks_pages,
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
