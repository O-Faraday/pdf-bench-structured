"""Run a parser on a PDF and keep the answer on disk, so a paid or slow service is not called twice.

``results/<dataset path>/<parser>/<pdf stem>.<key>.json`` (e.g. ``results/data/structured_whole_model/claude/x.ab12.json``;
the dataset path is the PDF's folder, relative to the working directory when it is inside it) holds the pages, the raw service response when there is one,
the options used and the time taken. The key changes when the PDF or the parser's options change.
Failures are not cached.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path

from .parsers import Parser, lookup

DEFAULT_DIR = Path("results")


@contextlib.contextmanager
def settings(parser: Parser, options: dict[str, str] | None):
    """Apply a variant's settings (``PDF_BENCH_<PARSER>_<KEY>``) for the duration of a run, then restore."""
    names = {k: f"PDF_BENCH_{parser.name}_{k}".upper().replace("-", "_") for k in (options or {})}
    before = {env: os.environ.get(env) for env in names.values()}
    try:
        for k, v in (options or {}).items():
            if v != "":
                os.environ[names[k]] = v
        yield
    finally:
        for env, old in before.items():
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old


def cache_key(parser: Parser, pdf: Path) -> str:
    options = {k: parser.option(k) for k in sorted(parser.options)}
    blob = pdf.read_bytes() + json.dumps([parser.name, parser.version, options], sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def dataset_dir(pdf: Path) -> Path:
    """The PDF's folder as a relative path: ``data/structured_whole_model``. A folder outside the working
    directory keeps its full path (without the drive or root), so two datasets never share answers."""
    folder = Path(pdf).resolve().parent
    try:
        return folder.relative_to(Path.cwd().resolve())
    except ValueError:
        return Path(*(part.rstrip(":\\/") for part in folder.parts[1:] if part not in ("/", "\\")))


def cache_path(parser: Parser, pdf: Path, cache_dir: Path = DEFAULT_DIR) -> Path:
    return Path(cache_dir) / dataset_dir(pdf) / parser.name / f"{pdf.stem}.{cache_key(parser, pdf)}.json"


def load_cached(name: str, pdf: str | Path, cache_dir: Path = DEFAULT_DIR, options: dict[str, str] | None = None,
                label: str | None = None) -> dict | None:
    parser = lookup(name)
    with settings(parser, options):
        path = cache_path(parser, Path(pdf), cache_dir)
    if path.is_file():
        return {**json.loads(path.read_text(encoding="utf-8")), "cached": True, "label": label or name}
    return None


def run(name: str, pdf: str | Path, *, cache_dir: Path = DEFAULT_DIR, force: bool = False,
        options: dict[str, str] | None = None, label: str | None = None) -> dict:
    """``{"label", "parser", "pdf", "pages", "raw", "seconds", "error", "options", "cached", "created"}``.

    ``options`` are the settings of a named variant; ``label`` is the name it is shown under.
    """
    pdf = Path(pdf)
    if not force and (hit := load_cached(name, pdf, cache_dir, options, label)):
        return hit
    parser = lookup(name)
    with settings(parser, options):
        res = parser.parse(pdf)
        out = {"label": label or name, "parser": name, "pdf": pdf.name, "pages": res.pages, "raw": res.raw,
               "seconds": round(res.seconds, 2), "error": res.error,
               "options": {k: parser.option(k) for k in sorted(parser.options)},
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "cached": False}
        path = cache_path(parser, pdf, cache_dir)
    if res.error is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    return out
