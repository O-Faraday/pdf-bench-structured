"""Adapters for the candidates of :mod:`pdf_bench.catalog`.

Three kinds:

* cloud document services (Databricks ``ai_parse_document``, LandingAI ADE): the PDF is uploaded and
  the service returns typed elements, which are written out as Markdown with HTML tables;
* command line tools (Marker, MinerU, olmOCR, Chandra): run in their own environment through their
  CLI, so their heavy dependencies never clash with this project's;
* page-image models: the PDF is rendered page by page and each image goes to a hosted vision model
  (Claude, Gemini, GPT);
* Mistral OCR, which takes the PDF itself.

Nothing here is exercised against the real services in this repository's tests (no network, no GPU,
no credentials): the tests use fake clients, and every adapter is marked ``tested=False`` in the
catalogue until it has been run for real.
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Callable
from pathlib import Path

from . import config

PAGE_PROMPT = (
    "Convert this document page to Markdown.\n"
    "- Keep the natural reading order, including across columns.\n"
    "- Render every table as an HTML <table> with <th> header cells and rowspan/colspan where needed.\n"
    "- Write formulas as LaTeX.\n"
    "- Leave out running page headers, footers and page numbers.\n"
    "- Do not describe images; write [image] where a figure is, then its caption as plain text.\n"
    "- Keep headings as Markdown headings with the right level, lists as lists (with nesting), "
    "links as [text](url) and footnotes as [^n]: text.\n"
    "- Write a form field as 'Label: value' and a checkbox as ☒ label (ticked) or ☐ label (empty).\n"
    "Return only the converted content, with no comment."
)

class Pages(list):
    """Pages of Markdown that remember the service's raw response (shown next to the scores)."""

    raw: object = None


DEFAULT_DPI = 150
DEFAULT_MAX_PAGES = 30
MAX_TOKENS = 8192


# --- page images --------------------------------------------------------------------------------

def render_pages(path: Path, *, dpi: int = DEFAULT_DPI, max_pages: int = DEFAULT_MAX_PAGES) -> list[bytes]:
    """PNG bytes of each page (at most ``max_pages``)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        out = []
        for i in range(min(len(pdf), max_pages)):
            image = pdf[i].render(scale=dpi / 72).to_pil()
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out
    finally:
        pdf.close()


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


def _int(parser, key: str, default: int) -> int:
    try:
        return int(parser.option(key, str(default)))
    except ValueError:
        return default


def strip_fence(text: str) -> str:
    """Models often wrap the whole answer in a ```markdown fence: remove it."""
    t = text.strip()
    m = re.fullmatch(r"```(?:markdown|md|html)?\s*\n(.*?)\n```", t, flags=re.S)
    return (m.group(1) if m else t).strip()


# --- page-image models --------------------------------------------------------------------------

def _anthropic_client():
    import anthropic

    return anthropic.Anthropic(api_key=config.get("ANTHROPIC_API_KEY"))


def _openai_client(base_url: str | None = None, api_key: str | None = None):
    import openai

    # the OpenAI key is only sent to OpenAI itself, never to another server (local servers, other gateways...)
    key = api_key or (config.get("OPENAI_API_KEY") if not base_url else "") or "EMPTY"
    return openai.OpenAI(base_url=base_url or None, api_key=key)


def _gemini_client():
    from google import genai

    return genai.Client(api_key=config.get("GEMINI_API_KEY") or config.get("GOOGLE_API_KEY"))


def _complete(reason, ok: tuple, name: str, page: int) -> None:
    """A page cut off by the token limit (or refused) would be scored as if the model had missed what follows."""
    if reason not in ok:
        raise RuntimeError(f"page {page} of the answer is incomplete ({name} = {reason}): not scored")


def anthropic_pages(parser, path: Path) -> list[str]:
    client = _anthropic_client()
    model = parser.option("model")
    pages, used = Pages(), {"input_tokens": 0, "output_tokens": 0}
    for png in render_pages(path, dpi=_int(parser, "dpi", DEFAULT_DPI),
                            max_pages=_int(parser, "max_pages", DEFAULT_MAX_PAGES)):
        msg = client.messages.create(
            model=model, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(png)}},
                {"type": "text", "text": parser.option("prompt", PAGE_PROMPT)}]}])
        pages.append(strip_fence("".join(b.text for b in msg.content if getattr(b, "type", "") == "text")))
        _complete(getattr(msg, "stop_reason", None), ("end_turn", "stop_sequence", None), "stop_reason", len(pages))
        u = getattr(msg, "usage", None)
        used["input_tokens"] += getattr(u, "input_tokens", 0) or 0
        used["output_tokens"] += getattr(u, "output_tokens", 0) or 0
    pages.raw = {"usage": used}
    return pages


def openai_compat_pages(parser, path: Path, *, base_url: str | None = None, api_key: str | None = None) -> list[str]:
    """Chat-completions with an image per page: OpenAI itself, vLLM, Ollama, LM Studio..."""
    client = _openai_client(base_url or parser.option("base_url") or None, api_key)
    model = parser.option("model")
    limit = {"max_completion_tokens": MAX_TOKENS}
    pages, used = Pages(), {"input_tokens": 0, "output_tokens": 0}
    for png in render_pages(path, dpi=_int(parser, "dpi", DEFAULT_DPI),
                            max_pages=_int(parser, "max_pages", DEFAULT_MAX_PAGES)):
        resp = client.chat.completions.create(
            model=model, **limit,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": parser.option("prompt", PAGE_PROMPT)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(png)}"}}]}])
        pages.append(strip_fence(resp.choices[0].message.content or ""))
        _complete(getattr(resp.choices[0], "finish_reason", None), ("stop", None), "finish_reason", len(pages))
        u = getattr(resp, "usage", None)
        used["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        used["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
    pages.raw = {"usage": used}
    return pages


def gemini_pages(parser, path: Path) -> list[str]:
    from google.genai import types

    client = _gemini_client()
    model = parser.option("model")
    pages, used = Pages(), {"input_tokens": 0, "output_tokens": 0}
    for png in render_pages(path, dpi=_int(parser, "dpi", DEFAULT_DPI),
                            max_pages=_int(parser, "max_pages", DEFAULT_MAX_PAGES)):
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=png, mime_type="image/png"), parser.option("prompt", PAGE_PROMPT)])
        pages.append(strip_fence(resp.text or ""))
        fin = getattr(resp.candidates[0], "finish_reason", None) if getattr(resp, "candidates", None) else None
        _complete(getattr(fin, "name", fin), ("STOP", None), "finish_reason", len(pages))
        u = getattr(resp, "usage_metadata", None)
        used["input_tokens"] += getattr(u, "prompt_token_count", 0) or 0
        used["output_tokens"] += getattr(u, "candidates_token_count", 0) or 0
    pages.raw = {"usage": used}
    return pages


def mistral_ocr_pages(parser, path: Path) -> list[str]:
    from mistralai import Mistral

    client = Mistral(api_key=config.get("MISTRAL_API_KEY"))
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    kwargs = dict(model=parser.option("model", "mistral-ocr-latest"),
                  document={"type": "document_url", "document_url": f"data:application/pdf;base64,{data}"},
                  include_image_base64=False)
    try:
        resp = client.ocr.process(table_format="html", **kwargs)
    except TypeError:  # older SDK without table_format
        resp = client.ocr.process(**kwargs)
    return [p.markdown for p in sorted(resp.pages, key=lambda p: p.index)]


# --- cloud document services: typed elements -> Markdown -------------------------------------------------

_ANCHOR = re.compile(r"<a id=['\"][^'\"]*['\"]></a>\s*")


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
        deadline = time.time() + _int(parser, "timeout", 900)
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


_ADE_WRAP = re.compile(r"<::(.*?)::>", re.S)
_ADE_KIND = re.compile(r":\s*(figure|logo|card|attestation|scan_code)\s*$", re.I)
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(/|of)\s*\d+)?$", re.I)


def _ade_unwrap(markdown: str) -> str:
    """The text inside an image-based chunk's ``<::description: figure::>`` wrapper (the wrapper is not text)."""
    inner = [_ADE_KIND.sub("", m.strip()).strip() for m in _ADE_WRAP.findall(markdown)]
    outside = _ADE_WRAP.sub(" ", markdown).strip()
    return "\n\n".join(x for x in (outside, *inner) if x)


def _ade_running(chunks) -> set[str]:
    """Margin texts that repeat on several pages once the digits are ignored: the running headers and footers."""
    pages: dict[str, set[int]] = {}
    for ch in chunks:
        if getattr(ch, "type", "") == "marginalia":
            key = re.sub(r"\d+", "#", _ANCHOR.sub("", ch.markdown or "")).strip().lower()
            page = int(getattr(getattr(ch, "grounding", None), "page", 0) or 0)
            pages.setdefault(key, set()).add(page)
    return {k for k, v in pages.items() if len(v) >= 2}


def landingai_chunks_to_pages(chunks) -> list[str]:
    """Markdown per page from LandingAI ADE chunks (``id``, ``type``, ``markdown``, ``grounding.page``).

    A ``marginalia`` chunk (the margins of the page) is left out only when it is a page number or repeats on
    several pages (running header or footer); any other one is kept, because a footnote at the foot of the
    page is margin text too. Image-based chunks arrive as ``<::description: figure::>``: a figure keeps its
    description (the caption is part of the chunk, see the ADE markdown response page) after an ``[image]``
    marker, a signature block (``attestation``) gives its readable text and a ``[signature]`` marker.
    """
    by_page: dict[int, list[str]] = {}
    running = _ade_running(chunks)
    for ch in chunks:
        page = int(getattr(getattr(ch, "grounding", None), "page", 0) or 0)
        by_page.setdefault(page, [])
        kind = getattr(ch, "type", "")
        md = _ANCHOR.sub("", ch.markdown or "").strip()
        if kind == "marginalia":
            key = re.sub(r"\d+", "#", md).lower()
            if _PAGE_NUMBER.match(md.strip()) or key in running:
                continue
            text = md
        elif kind == "attestation":
            m = re.search(r"Readable Text:\s*(.+)", md)
            text = (m.group(1).strip() + "\n\n" if m else "") + "[signature]"
        elif kind in ("figure", "logo", "card", "scan_code"):
            text = "[image]" + (("\n\n" + d) if (d := _ade_unwrap(md)) else "")
        else:
            text = md
        if text:
            by_page[page].append(text)
    if not by_page:
        return []
    return ["\n\n".join(by_page.get(i, [])) for i in range(max(by_page) + 1)]


def landingai_pages(parser, path: Path) -> list[str]:
    from landingai_ade import LandingAIADE

    client = LandingAIADE(environment="eu") if parser.option("region", "us") == "eu" else LandingAIADE()
    resp = client.parse(document=path, model=parser.option("model", "dpt-2-latest"))
    pages = Pages(landingai_chunks_to_pages(resp.chunks) or [resp.markdown])
    pages.raw = resp.model_dump() if hasattr(resp, "model_dump") else None
    return pages


# --- more services: LlamaParse, Unstructured ---------------------------------------------------

def llamaparse_pages(parser, path: Path) -> list[str]:
    from llama_cloud import LlamaCloud

    tier = parser.option("tier", "agentic")
    if tier == "fast":
        raise ValueError("the 'fast' tier returns no Markdown: use cost_effective, agentic or agentic_plus")
    client = LlamaCloud(api_key=config.get("LLAMA_CLOUD_API_KEY"))
    file = client.files.create(file=Path(path), purpose="parse")
    result = client.parsing.parse(file_id=file.id, tier=tier, version=parser.option("version", "latest"),
                                  expand=["markdown"])
    pages = Pages([p.markdown for p in result.markdown.pages])
    pages.raw = result.model_dump() if hasattr(result, "model_dump") else None
    return pages


def _field(el, name, default=None):
    """Attribute or key of an element: the library gives objects, a saved response gives dicts."""
    return el.get(name, default) if isinstance(el, dict) else getattr(el, name, default)


def unstructured_elements_to_pages(elements) -> list[str]:
    """Markdown/HTML per page from ``unstructured`` elements.

    ``Title`` becomes a heading (level = ``category_depth`` + 1), consecutive ``ListItem`` a bullet list,
    ``Table`` its ``text_as_html``, ``Image`` ``[image]``. Running ``Header``, ``Footer`` and ``PageNumber`` are
    left out. The library has no footnote element, so footnotes arrive as ordinary text.
    """
    by_page: dict[int, list[str]] = {}
    items: list[str] = []  # a run of list items, flushed into the current page
    items_page = 1

    def flush():
        if items:
            by_page.setdefault(items_page, []).append("\n".join(items))
            items.clear()

    for el in elements:
        kind = _field(el, "category") or type(el).__name__
        text = (_field(el, "text") or "").strip()
        meta = _field(el, "metadata") or {}
        page = int(_field(meta, "page_number") or 1)
        if kind != "ListItem":
            flush()
        if kind in ("Header", "Footer", "PageNumber", "PageBreak"):
            continue
        if kind == "ListItem":
            if items and page != items_page:
                flush()
            items_page = page
            if text:
                items.append(f"- {text}")
            continue
        if kind == "Title":
            depth = _field(meta, "category_depth")
            out = f"{'#' * (min(int(depth or 0), 5) + 1)} {text}" if text else ""
        elif kind == "Table":
            out = _field(meta, "text_as_html") or text
        elif kind in ("Image", "Figure"):
            out = "[image]"
        else:  # NarrativeText, FigureCaption, Formula, Address, UncategorizedText...
            out = text
        if out:
            by_page.setdefault(page, []).append(out)
    flush()
    if not by_page:
        return []
    return ["\n\n".join(by_page.get(i, [])) for i in range(1, max(by_page) + 1)]


def unstructured_pages(parser, path: Path) -> list[str]:
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=str(path), strategy=parser.option("strategy", "hi_res"),
                             infer_table_structure=True)
    pages = Pages(unstructured_elements_to_pages(elements))
    pages.raw = [e.to_dict() for e in elements if hasattr(e, "to_dict")]
    return pages


# --- OpenDataLoader PDF (local, Java) -------------------------------------------------------------------------

def java_major() -> int | None:
    """Major version of the Java found on the PATH, or None."""
    import shutil
    import subprocess

    exe = shutil.which("java")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'version "(\d+)(?:\.(\d+))?', out.stderr + out.stdout)
    if not m:
        return None
    major = int(m.group(1))
    return int(m.group(2) or 0) if major == 1 else major  # "1.8.0" means Java 8


def opendataloader_body(html: str) -> str:
    """The <body> of the HTML document written by OpenDataLoader."""
    from .html_util import html_body

    return html_body(html)


ODL_PAGE = "<!--odl-page {}-->"
HYBRID_URL = "http://localhost:5002"
_hybrid_servers: dict[str, object] = {}  # url -> Popen of the servers started by this process


def _port_open(url: str) -> bool:
    import socket
    from urllib.parse import urlparse

    u = urlparse(url)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 80), timeout=2):
            return True
    except OSError:
        return False


def ensure_hybrid_server(url: str = HYBRID_URL, *, wait: float = 300.0) -> None:
    """Make sure an OpenDataLoader hybrid server answers at ``url``; start a local one when nothing does.

    Only a local address can be started (the port is taken from the URL). The server is stopped when this
    process ends. The first start loads the Docling models, which can take minutes (and downloads them once).
    """
    import atexit
    import shutil
    import subprocess
    import sys
    import time
    from urllib.parse import urlparse

    if _port_open(url):
        return
    u = urlparse(url)
    if (u.hostname or "") not in ("localhost", "127.0.0.1", "::1"):
        raise RuntimeError(f"no hybrid server answers at {url}: start it there, or leave hybrid_url empty to start one here")
    exe = shutil.which("opendataloader-pdf-hybrid") or str(Path(sys.executable).with_name(
        "opendataloader-pdf-hybrid.exe" if sys.platform == "win32" else "opendataloader-pdf-hybrid"))
    if not Path(exe).exists():
        raise RuntimeError('hybrid server not installed: uv add "opendataloader-pdf[hybrid]"')
    proc = subprocess.Popen([exe, "--port", str(u.port or 5002)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _hybrid_servers[url] = proc
    atexit.register(proc.terminate)
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the hybrid server stopped at start (exit code {proc.returncode}): run "
                               "`opendataloader-pdf-hybrid` in a terminal to see why")
        if _port_open(url):
            return
        time.sleep(1)
    raise RuntimeError(f"the hybrid server did not answer within {int(wait)} s")


def opendataloader_pages(parser, path: Path) -> list[str]:
    """OpenDataLoader PDF, local mode (or hybrid with a running hybrid server).

    One JVM per call. The HTML output is used by default because it keeps merged cells (rowspan/colspan);
    the Markdown formats are split by page; the HTML one is a single text.
    """
    import tempfile

    import opendataloader_pdf

    fmt = parser.option("format", "markdown_html")
    if fmt not in ("html", "markdown", "markdown_html"):
        raise ValueError("option 'format' must be markdown_html, html or markdown")
    kw: dict = {"image_output": "external", "quiet": True}
    if fmt != "html":  # the Markdown writer can mark the pages (the HTML one cannot); with HTML tables it keeps merged cells
        kw["markdown_page_separator"] = "\n" + ODL_PAGE.format("%page-number%") + "\n"
        kw["markdown_with_html"] = fmt == "markdown_html"  # without images the figures vanish from the output (written to the temp dir)
    hybrid = parser.option("hybrid", "")
    if hybrid:
        url = parser.option("hybrid_url") or HYBRID_URL
        ensure_hybrid_server(url)
        kw["hybrid"] = hybrid
        kw["hybrid_url"] = url
        if parser.option("hybrid_mode"):
            kw["hybrid_mode"] = parser.option("hybrid_mode")
    if str(parser.option("use_struct_tree", "")).lower() in ("1", "true", "yes"):
        kw["use_struct_tree"] = True
    if str(parser.option("include_header_footer", "")).lower() in ("1", "true", "yes"):
        kw["include_header_footer"] = True
    with tempfile.TemporaryDirectory() as out:
        opendataloader_pdf.convert(input_path=[str(path)], output_dir=out, format=f"{'html' if fmt == 'html' else 'markdown'},json", **kw)
        main = Path(out) / (path.stem + (".html" if fmt == "html" else ".md"))
        if not main.exists():
            raise RuntimeError(f"OpenDataLoader wrote no {fmt} file")
        text = main.read_text(encoding="utf-8")
        raw_file = Path(out) / (path.stem + ".json")
        raw = raw_file.read_text(encoding="utf-8") if raw_file.exists() else None
    if fmt == "html":
        pages = Pages([opendataloader_body(text)])
    else:
        parts = re.split(r"<!--odl-page \d+-->", text)
        pages = Pages([x.strip() for x in parts[1:]] if len(parts) > 1 else [text.strip()])
    if raw:
        import json

        pages.raw = json.loads(raw)
    return pages


# --- OpenDocRouter: one API for many document-parsing models --------------------------------------------------

ODR_URL = "https://www.opendocrouter.ai"
ODR_INLINE_MAX = 3_000_000  # bytes: larger files go through the upload endpoint (the body is capped at 4 MB)


def _odr_post(client, url: str, body: dict | None, *, tries: int = 4):
    """POST with a retry on 429/503 (``Retry-After`` honoured, at most 30 s)."""
    import time

    for attempt in range(tries):
        resp = client.post(url, json=body) if body is not None else client.post(url)
        if resp.status_code in (429, 503) and attempt < tries - 1:
            time.sleep(min(float(resp.headers.get("Retry-After", 5) or 5), 30))
            continue
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error", {})
            except ValueError:
                err = {}
            raise RuntimeError(f"OpenDocRouter {resp.status_code} {err.get('code', '')}: {err.get('message', resp.text[:200])}")
        return resp.json()
    raise RuntimeError("OpenDocRouter: out of retries")  # pragma: no cover


def opendocrouter_pages(parser, path: Path) -> list[str]:
    """Parse with ``POST /v1/parse`` (sync): the document goes inline (<= 3 MB) or through an upload, in
    requests of at most ``pages_per_request`` pages. Any failed page raises (failed pages are not billed, and
    a partial answer would be scored as if the model had missed them). The raw responses keep the usage and
    ``charge_usd`` of every request."""
    import httpx

    path = Path(path)
    data = path.read_bytes()
    base = (parser.option("base_url") or ODR_URL).rstrip("/")
    model = parser.option("model")
    step = _int(parser, "pages_per_request", 50)
    n_pages = _pdf_page_count(path)
    ranges = ([None] if n_pages <= step else [f"{a}-{min(a + step - 1, n_pages)}" for a in range(1, n_pages + 1, step)])
    out: dict[int, str] = {}
    failed: list[str] = []
    responses = []
    headers = {"Authorization": f"Bearer {config.get('OPEN_DOC_ROUTER_API_KEY')}"}
    with httpx.Client(base_url=base, headers=headers, timeout=_int(parser, "timeout", 330)) as client:
        for pages in ranges:
            if len(data) <= ODR_INLINE_MAX:
                document = {"data": base64.b64encode(data).decode(), "mime_type": "application/pdf"}
            else:  # an upload is single-use: one per request
                up = _odr_post(client, "/v1/uploads", None)
                put = httpx.put(up["upload_url"], content=data, timeout=300)
                put.raise_for_status()
                document = {"upload_id": up["upload_id"]}
            body = {"model": model, "document": document, **({"pages": pages} if pages else {})}
            resp = _odr_post(client, "/v1/parse", body)
            responses.append(resp)
            for pg in resp.get("pages", []):
                if pg.get("status") == "ok":
                    out[int(pg["page"])] = pg.get("markdown", "")
                else:
                    err = pg.get("error") or {}
                    failed.append(f"p{pg.get('page')} {err.get('code', 'error')}")
    if failed:
        raise RuntimeError(f"{len(failed)} page(s) failed (not billed): " + ", ".join(failed[:10]))
    pages_md = Pages([out[k] for k in sorted(out)])
    pages_md.raw = {"charge_usd": round(sum(float(r.get("charge_usd") or 0) for r in responses), 6),
                    "model_version": responses[0].get("model_version") if responses else None, "requests": responses}
    return pages_md


def _pdf_page_count(path: Path) -> int:
    import pymupdf

    with pymupdf.open(path) as doc:
        return len(doc)
