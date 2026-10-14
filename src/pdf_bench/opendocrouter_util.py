"""The models OpenDocRouter serves, with their published ParseBench scores and prices (public endpoint, no key).

ParseBench is run and published by the service itself: treat the scores as a prior, not as a verdict.
"""

from __future__ import annotations

import json
import urllib.request

BASE_URL = "https://www.opendocrouter.ai"
SCORES = ["score", "tables", "charts", "content_faithfulness", "semantic_formatting", "visual_grounding"]
SCORE_LABELS = {"score": "overall", "tables": "tables", "charts": "charts", "content_faithfulness": "faithfulness",
                "semantic_formatting": "formatting", "visual_grounding": "grounding"}


def fetch(base_url: str = BASE_URL, timeout: int = 20) -> dict:
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/models", headers={"User-Agent": "pdf-bench"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https URL
        return json.loads(resp.read().decode("utf-8"))


def models(payload: dict | None = None, *, rank_by: str = "score") -> list[dict]:
    """Rows sorted by a ParseBench score (best first); models not benchmarked yet come last."""
    payload = payload if payload is not None else fetch()
    rows = []
    for m in payload.get("data", []):
        bench = m.get("parsebench") or {}
        avg = m.get("avg_charge_per_page_usd")
        rows.append({"id": m["id"], "name": m.get("name", m["id"]), "version": m.get("version"),
                     **{SCORE_LABELS[k]: bench.get(k) for k in SCORES},
                     "usd_per_1k_pages": round(avg * 1000, 2) if avg is not None else None,
                     "max_sync_pages": m.get("max_sync_pages")})
    key = SCORE_LABELS.get(rank_by, rank_by)
    return sorted(rows, key=lambda r: (r.get(key) is None, -(r.get(key) or 0), r["id"]))


def label(model_id: str) -> str:
    """Short name used in the tables: ``google/gemini-3-flash`` -> ``odr:gemini-3-flash``."""
    return "odr:" + model_id.split("/", 1)[-1]
