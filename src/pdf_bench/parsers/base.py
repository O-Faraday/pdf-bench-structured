"""The registry and the parser class: what every parser module in this package builds on.

Each parser turns a PDF into one string per page (Markdown or HTML) and is timed. Only parsers that
recover document structure (headings, tables, lists...) are listed; plain text extractors are out of
scope. A parser whose library, tool or credentials are missing reports why instead of crashing a run.
One module per parser sits next to this one; ``__init__`` imports them all.
"""

from __future__ import annotations

import functools
import importlib.util
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

CHECKED = "2026-10-07"

FAMILIES = {
    "library": "Libraries and pipelines (local)",
    "hosted_vlm": "Hosted models (documents leave your machine, paid)",
    "cloud_service": "Cloud document services",
}

REGISTRY: dict[str, "Parser"] = {}
# The oracles read the ground truth: they are score ceilings, not parsers, so they live apart from REGISTRY.
REFERENCES: dict[str, "Parser"] = {}


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
        from .. import config

        env = f"PDF_BENCH_{self.name}_{key}".upper().replace("-", "_")
        return config.get(env) or str(self.options.get(key, default))

    def status(self) -> str:
        """Empty when the parser can run here, otherwise why it cannot (shown in the app)."""
        from .. import config

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


def register(parser: Parser) -> Parser:
    REGISTRY[parser.name] = parser
    return parser


def lookup(name: str) -> Parser:
    """A parser or a reference by name."""
    return REGISTRY[name] if name in REGISTRY else REFERENCES[name]


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


def entry(name: str, label: str, family: str, adapter, **kw) -> Parser:
    """Build and register a parser. ``adapter`` is called as adapter(parser, path); None = listed, not implemented."""
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
        p.check = check(functools.partial(REGISTRY.__getitem__, name))
    return register(p)


def reference(name: str, *args, **kw) -> Parser:
    """A reference (oracle): built like a parser, kept out of the parser list."""
    p = entry(name, *args, **kw)
    REFERENCES[name] = REGISTRY.pop(name)
    return p


def any_env(*names: str):
    """Readiness: at least one of the environment variables is set."""

    def factory(_getter):
        def check() -> str:
            from .. import config

            return "" if any(config.get(n) for n in names) else "missing in .env: " + " or ".join(names)

        return check

    return factory


def chain(*checks):
    def check() -> str:
        for c in checks:
            if reason := c():
                return reason
        return ""

    return check


def model_required(_getter):
    def check() -> str:
        p = _getter()
        return "" if p.option("model") else f"set PDF_BENCH_{p.name.upper()}_MODEL (or the model setting in the app) to the model id"

    return check
