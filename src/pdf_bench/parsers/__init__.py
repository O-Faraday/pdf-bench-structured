"""The parsers: one module per parser, registered in the order of this list (the order of the app).

``base`` holds the registry and the :class:`Parser` class, ``common`` what several parsers share.
Adding a parser = adding a module here that calls ``entry(...)`` and importing it below.
"""

# isort: skip_file
from __future__ import annotations

from .base import (CHECKED, FAMILIES, REFERENCES, REGISTRY, ParseResult, Parser, available_parsers, lookup, register,
                   run_parsers, unavailable, unknown)

# The order of these imports is the order of REGISTRY, which is the order of the app.
from . import pymupdf4llm, docling  # noqa: F401,E402  (local libraries)
from . import claude, gemini, gpt, opendocrouter, mistral_ocr  # noqa: F401,E402  (hosted vision models)
from . import databricks, landingai, llamaparse  # noqa: F401,E402  (cloud document services)
from . import unstructured, opendataloader  # noqa: F401,E402  (more local libraries)
from . import oracles  # noqa: F401,E402  (references, not parsers)

__all__ = ["CHECKED", "FAMILIES", "REFERENCES", "REGISTRY", "ParseResult", "Parser", "available_parsers", "lookup",
           "register", "run_parsers", "unavailable", "unknown"]
