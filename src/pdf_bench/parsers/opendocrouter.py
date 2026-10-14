"""OpenDocRouter: one parse endpoint for document-parsing specialists and frontier models."""

from __future__ import annotations

import base64
from pathlib import Path

from .. import config
from .base import entry, model_required
from .common import Pages, int_option

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
    step = int_option(parser, "pages_per_request", 50)
    n_pages = _pdf_page_count(path)
    ranges = ([None] if n_pages <= step else [f"{a}-{min(a + step - 1, n_pages)}" for a in range(1, n_pages + 1, step)])
    out: dict[int, str] = {}
    failed: list[str] = []
    responses = []
    headers = {"Authorization": f"Bearer {config.get('OPEN_DOC_ROUTER_API_KEY')}"}
    with httpx.Client(base_url=base, headers=headers, timeout=int_option(parser, "timeout", 330)) as client:
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


entry("opendocrouter", "OpenDocRouter (one API, many parsing models)", "hosted_vlm", opendocrouter_pages,
       requires="httpx", needs=("OPEN_DOC_ROUTER_API_KEY",), hosted=True,
       options={"model": "", "pages_per_request": 50}, check=lambda g: model_required(g),
       install="included", version="the model id you give", license="depends on the model",
       source="https://www.opendocrouter.ai/",
       note="Made by LlamaIndex. Serves document-parsing specialists (MinerU2.5-Pro, PaddleOCR-VL, LightOnOCR, "
            "dots.mocr...) and frontier models behind one parse endpoint; only successful pages are billed. "
            "The app lists the models with their ParseBench scores (published by the service itself): make one "
            "named variant per model id (Models tab). Documents leave your machine. 'pages_per_request' must not "
            "exceed the model's max_sync_pages")
