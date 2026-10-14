"""Scoring as the app and the analysis files show it: post-processing, then the scorer, then "not applicable".

``frame()`` is the one place that turns a parser's pages into scored rows, so the Compare page, the
Documents and models page, the Compare page and the analysis files always agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import judge, locate, postprocess
from .elements import load_gt
from .parsers import lookup
from .score import score


def frame(label: str, parser: str, pdf: str | Path, pages: list[str], steps: list[str] | tuple[str, ...] = ()) -> pd.DataFrame:
    """Scored rows of one answer. Structure is NaN (not applicable) for the kinds the tool cannot express."""
    pdf = Path(pdf)
    out = postprocess.apply(list(pages), list(steps)) if steps else list(pages)
    gt = locate.annotate(load_gt(pdf.with_suffix(".gt.json")), pdf)  # pages read back from the PDF
    f = score(gt, out, parser=label, raw_pages=list(pages)).frame()
    cannot = {k: v for k, v in lookup(parser).cannot.items()
              if not any(k in postprocess.CREATES.get(x, ()) for x in steps)}  # a step now builds those from text
    na = f["kind"].isin(cannot)
    f["structure"] = f["structure"].astype(float)
    f.loc[na, "structure"] = np.nan
    f.loc[na, "detail"] = f.loc[na, "kind"].map(lambda k: f"not applicable: {cannot[k]}")
    return f


def not_applicable(parser: str) -> list[str]:
    return sorted(lookup(parser).cannot)


def judged(f: pd.DataFrame, parser: str, pages: list[str], steps: list[str] | tuple[str, ...] = (), **kw) -> pd.DataFrame:
    """``f`` (from ``frame``) with the LLM judge's verdict on each row, on the same text the scorer read.

    The judge's structure is left empty where the rules' structure is "not applicable", so the two columns
    are always compared on the same rows. Keyword arguments go to ``judge.judge_frame`` (``call``, ``force``, ``tally``...).
    """
    out = postprocess.apply(list(pages), list(steps)) if steps else list(pages)
    g = judge.judge_frame(f, out, **kw)
    g.loc[f["structure"].isna(), "judge_structure"] = np.nan
    return g
