"""List the models OpenRouter serves that accept images, so they can be picked in the app.

The model list is public (no key needed). Prices are per token in the API; they are shown per million tokens.
"""

from __future__ import annotations

import json
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"

# Starting point only: ids as listed by OpenRouter on 2026-10-09, kept if the live list still has them.
SUGGESTED = [
    "anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5", "openai/gpt-6.1-sol", "openai/gpt-6-luna",
    "mistralai/mistral-large-4-0", "qwen/qwen3.8-omni-flash", "z-ai/glm-5.3-flashx",
    "deepseek/deepseek-v4.1-flash", "inclusionai/ling-3.0-flash-vl",
]


def fetch(timeout: int = 20) -> dict:
    req = urllib.request.Request(MODELS_URL, headers={"User-Agent": "pdf-bench"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read().decode("utf-8"))


def vision_models(payload: dict | None = None, *, include_batch: bool = False) -> list[dict]:
    """Models that take images and return text: id, name, $ per million input / output tokens, context.

    Router entries (negative price) and ``:batch`` duplicates are left out unless asked for.
    """
    payload = payload if payload is not None else fetch()
    rows = []
    for m in payload.get("data", []):
        arch = m.get("architecture") or {}
        if "image" not in (arch.get("input_modalities") or []) or "text" not in (arch.get("output_modalities") or ["text"]):
            continue
        price = m.get("pricing") or {}
        try:
            p_in, p_out = float(price.get("prompt", -1)), float(price.get("completion", -1))
        except (TypeError, ValueError):
            continue
        if p_in < 0 or p_out < 0:
            continue
        if m["id"].endswith(":batch") and not include_batch:
            continue
        rows.append({"id": m["id"], "name": m.get("name", m["id"]), "in_per_m": round(p_in * 1e6, 3),
                     "out_per_m": round(p_out * 1e6, 3), "context": m.get("context_length")})
    return sorted(rows, key=lambda r: (r["in_per_m"], r["id"]))


def label(model_id: str) -> str:
    """Short name used in the tables: ``anthropic/claude-sonnet-5.5`` -> ``or:claude-sonnet-5.5``."""
    return "or:" + model_id.split("/", 1)[-1]
