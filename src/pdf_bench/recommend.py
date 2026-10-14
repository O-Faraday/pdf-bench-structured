"""Which models to advise: the three most reliable on structure, among the free ones and among the paid ones.

The ranking looks at what the structured output is used for downstream (text fields for chunking, other
elements linked to them): the structure score of **tables, figures and captions, form fields and checkboxes**
and the text read from **scans** (OCR). The overall structure score only breaks ties. A model is ranked only if
it ran on every selected PDF without error. Nothing here reads a model's name: free or paid comes from the cost.
"""

from __future__ import annotations

import pandas as pd

FOCUS_KINDS = ("table", "table_continuity", "figure", "caption", "field", "checkbox")
TIE = 0.02  # differences below this are not told apart
# The model that wrote the test PDFs (the generator's reference, close to a training set for it). It is run and
# shown as a reference point, never ranked or advised: a score it earns on documents it made says nothing about
# how it reads documents it did not.
REFERENCE_MODELS = ("claude",)


def _mean(values) -> float | None:
    v = pd.Series(list(values), dtype="float64").dropna()
    return round(float(v.mean()), 3) if len(v) else None


def focus_scores(frame: pd.DataFrame, scan_pdfs: set[str]) -> dict:
    """``frame``: every scored row of one model (columns pdf, kind, content, structure)."""
    if frame.empty:
        return {"focus": None, "ocr": None, "overall": None, "weak": []}
    per_kind = frame.groupby("kind")[["content", "structure"]].mean()
    kinds = [k for k in FOCUS_KINDS if k in per_kind.index and pd.notna(per_kind.loc[k, "structure"])]
    focus = _mean(per_kind.loc[k, "structure"] for k in kinds)
    ocr_rows = frame[frame["pdf"].isin(scan_pdfs)]
    ocr = _mean(ocr_rows.groupby("kind")["content"].mean()) if len(ocr_rows) else None
    parts = [x for x in (focus, ocr) if x is not None]
    weak = sorted((k for k in kinds if per_kind.loc[k, "structure"] < 0.8), key=lambda k: per_kind.loc[k, "structure"])
    return {"focus": _mean(parts), "ocr": ocr, "structure on tables, figures, forms": focus,
            "overall": _mean(per_kind["structure"]), "weak": weak[:3]}


def advise(models: list[dict]) -> dict:
    """``models``: one dict per model with name, free (bool), cost (USD or None), seconds, runs, expected_runs,
    errors, not_applicable (list), and the output of :func:`focus_scores`.

    Returns ``{"free": [...], "paid": [...], "excluded": [(name, reason)], "notes": [str]}`` where each ranked
    entry is the model dict plus ``rank`` and ``why``.
    """
    ranked, excluded = {"free": [], "paid": []}, []
    for m in models:
        if m["name"] in REFERENCE_MODELS:
            excluded.append((m["name"], "reference: it wrote these PDFs, so it is not ranked"))
        elif m["focus"] is None:
            excluded.append((m["name"], "no score"))
        elif m["errors"] or m["runs"] < m["expected_runs"]:
            excluded.append((m["name"], f"{m['runs']}/{m['expected_runs']} PDFs scored, {m['errors']} error(s)"))
        else:
            ranked["free" if m["free"] else "paid"].append(m)
    out = {"excluded": excluded, "notes": []}
    for cat, items in ranked.items():
        items.sort(key=lambda m: (-m["focus"], -(m["overall"] or 0), m["cost"] if m["cost"] is not None else 1e9, m["seconds"]))
        top = items[:3]
        for i, m in enumerate(top, 1):
            m["rank"] = i
            m["why"] = _why(m)
        out[cat] = top
        if len(top) >= 2 and top[0]["focus"] - top[1]["focus"] < TIE:
            out["notes"].append(f"{cat}: {top[0]['name']} and {top[1]['name']} are within {TIE} of each other on the "
                                "focus score: do not read the order as a real difference on this few PDFs.")
    if not ranked["paid"]:
        out["notes"].append("No paid model was run (hosted models and cloud services run only when ticked): "
                            "the paid list is empty, not an absence of candidates.")
    return out


def _why(m: dict) -> str:
    bits = [f"focus {m['focus']:.2f}"]
    if m.get("ocr") is not None:
        bits.append(f"scan text {m['ocr']:.2f}")
    if m["weak"]:
        bits.append("weak on " + ", ".join(m["weak"]))
    if m["not_applicable"]:
        bits.append("cannot express " + ", ".join(m["not_applicable"]))
    bits.append("free" if m["free"] else (f"~${m['cost']:.4f} for these PDFs" if m["cost"] is not None else "cost unknown"))
    bits.append(f"{m['seconds']:.0f} s")
    return "; ".join(bits)
