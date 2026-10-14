"""Small helpers for HTML written by parsers."""

from __future__ import annotations

import re


def html_body(html: str) -> str:
    """The content of <body> (a full HTML document has a head, styles and scripts the scorer must not read)."""
    m = re.search(r"<body[^>]*>(.*)</body>", html, flags=re.S | re.I)
    return (m.group(1) if m else re.sub(r"<(style|script)[^>]*>.*?</\1>", "", html, flags=re.S | re.I)).strip()
