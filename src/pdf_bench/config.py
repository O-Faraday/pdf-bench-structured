"""Secrets and settings read from a ``.env`` file (or the environment).

``.env`` is searched from the current folder upwards. Values already set in the environment win over
the file. Nothing in this project prints, logs or stores a secret.

Variables used by the parsers (see :mod:`pdf_bench.parsers`): ``HF_TOKEN`` (model downloads),
``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``, ``MISTRAL_API_KEY``,
``VISION_AGENT_API_KEY`` (LandingAI), ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` /
``DATABRICKS_WAREHOUSE_ID``, and ``PDF_BENCH_<PARSER>_<OPTION>`` for per-parser options.
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded: Path | None | bool = False


def load_env(path: str | Path | None = None) -> Path | None:
    """Load ``.env`` once. Returns the file that was read, or ``None`` if there is none."""
    global _loaded
    from dotenv import find_dotenv, load_dotenv

    if path is None and _loaded is not False:
        return _loaded  # type: ignore[return-value]
    found = Path(path) if path else (Path(f) if (f := find_dotenv(usecwd=True)) else None)
    if found and found.is_file():
        load_dotenv(found, override=False)
        result: Path | None = found
    else:
        result = None
    if path is None:
        _loaded = result
    return result


def get(name: str, default: str | None = None) -> str | None:
    """A variable from the environment or ``.env``; empty values count as unset."""
    load_env()
    value = os.environ.get(name, "").strip()
    return value or default
