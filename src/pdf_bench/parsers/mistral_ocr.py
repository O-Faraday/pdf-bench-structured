"""Mistral OCR: takes the PDF itself."""

from __future__ import annotations

import base64
from pathlib import Path

from .. import config
from .base import entry


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


entry("mistral_ocr", "Mistral OCR", "hosted_vlm", mistral_ocr_pages, requires="mistralai",
       needs=("MISTRAL_API_KEY",), hosted=True, options={"model": "mistral-ocr-latest"},
       install="uv add mistralai", version="mistral-ocr-latest (alias)", license="commercial API",
       source="https://docs.mistral.ai", reference="OmniDocBench v1.6: 85.66; olmOCR-Bench: 72.0",
       note="The alias always points to the newest Mistral OCR model; the response is split by page")
