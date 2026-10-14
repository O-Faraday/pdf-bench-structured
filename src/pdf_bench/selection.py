"""The choice made in the app: documents to create, parsers to compare, their options.

Saved as ``benchmark.json`` so the scoring stage (and the command line) can reuse it. Secrets are never
stored: only names and non-secret options (model ids, URLs, volume paths).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import NamedTuple
from pathlib import Path

from .parsers import REFERENCES, REGISTRY, unavailable, unknown

DEFAULT_PATH = Path("benchmark.json")
SECRET_WORDS = ("token", "key", "secret", "password")


class Entry(NamedTuple):
    """One thing to run and to name in the tables: a parser with a given set of settings."""

    label: str
    parser: str
    options: dict[str, str]  # overrides on top of the parser's settings; empty = the saved/default settings


@dataclass
class Selection:
    parsers: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)  # "parser.key" -> value
    allow_hosted: bool = False
    docs_dir: str = "data/structured"
    docs_count: int | None = None
    docs_seed: int = 0
    docs_only: list[str] = field(default_factory=list)
    variants: dict[str, dict] = field(default_factory=dict)  # label -> {"parser": name, "options": {key: value}}
    docs_selected: list[str] = field(default_factory=list)  # PDF file names to run on; empty = every PDF of the folder
    postprocess: list[str] = field(default_factory=list)  # steps of pdf_bench.postprocess applied before scoring

    def entries(self) -> list[Entry]:
        """What to run, in order: each selected parser under its own name, then the named variants."""
        out = [Entry(n, n, {}) for n in self.parsers]
        out += [Entry(label, v.get("parser", ""), dict(v.get("options", {}))) for label, v in self.variants.items()]
        return out

    def problems(self) -> list[str]:
        out = []
        for label, v in self.variants.items():
            parser = REGISTRY.get(v.get("parser", ""))
            if label in REGISTRY or label in REFERENCES or label in self.parsers:
                out.append(f"variant '{label}': the name is already a parser name")
            if parser is None:
                out.append(f"variant '{label}': unknown parser '{v.get('parser')}'")
                continue
            for key in v.get("options", {}):
                if any(w in key.lower() for w in SECRET_WORDS):
                    out.append(f"variant '{label}': {key} looks like a secret: put it in .env, not here")
                elif key not in parser.options and key != "base_url":
                    out.append(f"variant '{label}': {parser.name} has no setting '{key}'")
        if not self.parsers and not self.variants:
            out.append("select at least one parser")
        if bad := unknown(self.parsers):
            out.append(f"unknown parser(s): {', '.join(bad)}")
        known = [n for n in self.parsers + [v.get("parser", "") for v in self.variants.values()] if n in REGISTRY]
        out += self._not_ready()
        if (hosted := [n for n in known if REGISTRY[n].hosted]) and not self.allow_hosted:
            out.append(f"{', '.join(hosted)} send the PDFs to a third party and cost money: confirm it")
        for key in self.options:
            if any(w in key.lower() for w in SECRET_WORDS):
                out.append(f"option {key} looks like a secret: put it in .env, not here")
        return out

    def _not_ready(self) -> list[str]:
        """Why a selected parser cannot run, judged with the settings of this selection (a base URL typed here counts)."""
        from .results import settings

        out = []
        for e in self.entries():
            parser = REGISTRY.get(e.parser)
            if parser is None:
                continue
            base = {k.partition(".")[2]: v for k, v in self.options.items() if k.partition(".")[0] == e.parser}
            with settings(parser, {**base, **e.options}):
                why = parser.status()
            if why:
                out.append(f"{e.label}: {why}")
        return out

    def apply(self) -> None:
        """Make the options visible to the parsers (``PDF_BENCH_<PARSER>_<KEY>``) for this process."""
        for key, value in self.options.items():
            parser, _, opt = key.partition(".")
            if value != "":
                os.environ[f"PDF_BENCH_{parser}_{opt}".upper().replace("-", "_")] = value

    def save(self, path: str | Path = DEFAULT_PATH) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False), encoding="utf-8")
        return path


def load(path: str | Path = DEFAULT_PATH) -> Selection:
    path = Path(path)
    if not path.is_file():
        return Selection()
    data = json.loads(path.read_text(encoding="utf-8"))
    return Selection(**{k: v for k, v in data.items() if k in Selection.__dataclass_fields__})
