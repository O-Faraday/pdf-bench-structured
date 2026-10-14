"""The references (oracles): the ground truth written as HTML or Markdown. Score ceilings, not parsers."""

from __future__ import annotations

from ..oracle import oracle_html_pages, oracle_markdown_pages
from .base import reference

reference("oracle_html", "Oracle (ground truth as HTML)", "reference", oracle_html_pages, tested=True,
       install="included", version="-", license="-",
       note="Writes the ground truth as HTML. Scores 1.0 by construction: it proves the scorer, and shows what "
            "a perfect HTML answer looks like")
reference("oracle_markdown", "Oracle (ground truth as Markdown)", "reference", oracle_markdown_pages, tested=True,
       install="included", version="-", license="-",
       note="Writes the ground truth as GitHub Markdown: the ceiling of Markdown, which cannot express merged "
            "or nested cells")
