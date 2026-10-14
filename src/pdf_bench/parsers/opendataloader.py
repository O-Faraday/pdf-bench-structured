"""OpenDataLoader PDF: a Java engine, local by default, with an optional hybrid server."""

from __future__ import annotations

import re
from pathlib import Path

from .base import entry
from .common import Pages


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


def _java_ready(_getter):
    def check() -> str:
        major = java_major()
        if major is None:
            return "Java not found on the PATH: install Java 11 or later (e.g. Temurin) and reopen the terminal"
        if major < 11:
            return f"Java {major} found: OpenDataLoader needs Java 11 or later"
        return ""

    return check


entry("opendataloader", "OpenDataLoader PDF", "library", opendataloader_pages, requires="opendataloader_pdf",
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
