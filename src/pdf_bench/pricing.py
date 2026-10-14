"""Estimated cost of a run, per model.

Local tools cost nothing but compute. Services bill per page (a fixed rate, or credits per page) or per
token (hosted vision models). Rates are public list prices read on ``CHECKED``; they change, so the
estimate is shown with its basis and the source. A model without a known rate has no estimate: it is
never guessed.

Token-billed models (Claude, Gemini, GPT): give the price in USD per million tokens with
``PDF_BENCH_<PARSER>_PRICE_IN_PER_M`` and ``PDF_BENCH_<PARSER>_PRICE_OUT_PER_M`` (``.env``). The token
counts come from the usage recorded with each answer (answers cached before this feature have none).
"""

from __future__ import annotations

from dataclasses import dataclass

CHECKED = "2026-10-09"

LOCAL = {"pymupdf4llm", "pymupdf4llm_max", "docling", "opendataloader", "unstructured"}
TOKEN_BILLED = {"claude", "gemini", "gpt"}


@dataclass(frozen=True)
class Rate:
    per_1000_pages: float  # USD
    basis: str
    source: str
    low: float | None = None  # a published range, when the price depends on the page
    high: float | None = None


MISTRAL = Rate(4.0, "OCR 4.1: $4 per 1,000 pages", "https://docs.mistral.ai/inference/pricing")
DATABRICKS = Rate(6.30, "ai_parse_document: $0.70-$6.30 per 1,000 pages by complexity (US East); upper bound used",
                  "https://www.databricks.com/product/pricing/ai-functions", 0.70, 6.30)
LLAMA_CREDITS = {"cost_effective": 3, "agentic": 10, "agentic_plus": 45}
LLAMA_USD_PER_1000_CREDITS = 1.25
LLAMA_SOURCE = "https://developers.llamaindex.ai/llamaparse/general/pricing/"
LANDING = Rate(30.0, "Parse: 3 credits per page, $0.01 per credit (Explore and Team plans; the rate of model dpt-2 is not listed)",
               "https://docs.landing.ai/ade/ade-credit-consumption.md")


def page_rate(name: str, option) -> Rate | None:
    """The per-page rate of a service, or None. ``option(key, default)`` reads the model's settings."""
    if name == "mistral_ocr":
        return MISTRAL
    if name == "databricks_ai_parse":
        return DATABRICKS
    if name == "landingai_ade":
        return LANDING
    if name == "llamaparse":
        tier = option("tier", "agentic")
        credits = LLAMA_CREDITS.get(tier)
        if credits is None:
            return None
        return Rate(credits * LLAMA_USD_PER_1000_CREDITS, f"tier {tier}: {credits} credits per page, "
                    f"${LLAMA_USD_PER_1000_CREDITS} per 1,000 credits", LLAMA_SOURCE)
    return None


@dataclass
class Cost:
    usd: float | None
    basis: str  # why this number (or why none)
    free: bool = False  # local tool: no per-use fee


def _float(text: str) -> float | None:
    try:
        return float(text) if str(text).strip() else None
    except ValueError:
        return None


def estimate(name: str, option, answers: list[dict], pages: int) -> Cost:
    """Cost of ``answers`` (the cached runs of one model) over ``pages`` PDF pages.

    ``option(key, default)`` reads the model's settings; ``pages`` is the number of pages of the PDFs run.
    """
    if name in LOCAL:
        return Cost(0.0, "free: runs on your machine", True)
    billed = [a["raw"]["charge_usd"] for a in answers if isinstance(a.get("raw"), dict) and "charge_usd" in a["raw"]]
    if billed:
        return Cost(round(sum(float(v) for v in billed), 4), "billed by the service (reported with the answers)")
    if name in TOKEN_BILLED:
        pin, pout = _float(option("price_in_per_m", "")), _float(option("price_out_per_m", ""))
        usage = [a["raw"]["usage"] for a in answers if isinstance(a.get("raw"), dict) and "usage" in a["raw"]]
        if pin is None or pout is None:
            return Cost(None, f"price not set: PDF_BENCH_{name.upper()}_PRICE_IN_PER_M and _PRICE_OUT_PER_M (USD per million tokens)")
        if len(usage) < len(answers) or not usage:
            return Cost(None, "no token usage recorded for some answers: run them again")
        tin = sum(u.get("input_tokens", 0) for u in usage)
        tout = sum(u.get("output_tokens", 0) for u in usage)
        return Cost(round((tin * pin + tout * pout) / 1e6, 4), f"{tin:,} tokens in x ${pin}/M + {tout:,} tokens out x ${pout}/M")
    rate = page_rate(name, option)
    if rate is None:
        return Cost(None, "no public rate known for this model")
    return Cost(round(rate.per_1000_pages * pages / 1000, 4), f"{pages} pages x ${rate.per_1000_pages}/1,000 pages. {rate.basis}")
