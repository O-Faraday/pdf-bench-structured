"""An LLM judge that answers, in words, the same questions the deterministic scorer answers with rules.

For every row of a score (a *question*: see ``questions.py``) the judge reads the text the scorer read (the parser
output after post-processing) and the expected element, and returns ``content`` and ``structure`` between 0
and 1 with a one-line reason. ``judge_content`` / ``judge_structure`` sit next to ``content`` / ``structure``
so the two methods can be compared: where they differ, either a rule is too strict or too lax, or the judge is wrong.
The reasons say which.

* The judge is text-only: it does not look at the PDF. The ground truth is the reference, as for the scorer.
* One call per (answer, group of questions): every question is judged on its own. Answers are cached by
  content (``results/judge/<hash>.json``), so re-evaluating costs nothing unless the output, the questions, the
  judge model or the prompt changed. ``PROMPT_VERSION`` must change with the instructions.
* Use a judge that is not one of the models under test, or say so: a model tends to favour its own style.

Settings (``.env`` or environment): ``PDF_BENCH_JUDGE_PROVIDER`` = ``anthropic`` (default) or ``openai``
(also any OpenAI-compatible server through ``PDF_BENCH_JUDGE_BASE_URL``), ``PDF_BENCH_JUDGE_MODEL``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, questions

PROMPT_VERSION = "3"
DEFAULT_DIR = Path("results") / "judge"
CHUNK = 20  # questions per call: keeps the answer short and each question in view
MAX_TOKENS = 6000
DEFAULTS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-6.1-sol"}

SYSTEM = """You are a strict, impartial evaluator of PDF-to-text extraction.

You get (1) the OUTPUT of an extraction tool for one PDF, written in Markdown/HTML, with page separators, and
(2) numbered QUESTIONS. Each question describes one element that the PDF really contains (the ground truth)
and says how to score it. Decide, from the OUTPUT only, how well that element was recovered.

For every question give:
- "content": 0 to 1, is the text of the element present in the OUTPUT, wherever it is? Allow differences of
  case, Markdown/HTML syntax, line breaks and trivial OCR slips; do not allow missing or altered words and numbers.
  Use null when the question says content has no meaning.
- "structure": 0 to 1, was it recovered as the right kind of element, in the right place, as the question's
  scoring rule says? Use null when the question says structure has no meaning.
- "reason": at most 20 words, naming the evidence you saw in the OUTPUT.

Content and structure are separate. "content" only asks whether the words are in the OUTPUT: a lost link target, a missing
tick mark, a heading level, a cell position, or text inside an image that was not extracted never lowers "content".

Rules: judge only what is written in the OUTPUT; never give credit for what is plausible; the expected text is the
reference, not the output. A tool that wrote the text as plain text when a heading, list item, table, link or
footnote was expected gets structure 0 (but keeps its content). Judge each question independently of the others.

Answer with a JSON array only, no prose, no code fence: [{"id": 1, "content": 1, "structure": 0, "reason": "..."}, ...]
with one object per question."""


ARBITER_VERSION = "1"
ARBITER_CHUNK = 10
ARBITER = """You settle disagreements between two evaluators of PDF-to-text extraction: a deterministic scorer (RULES) and an LLM
judge (JUDGE) who scored the same questions independently.

You get the OUTPUT of an extraction tool, then CASES. Each case gives the expected element, how it is scored, the DISPUTED
dimension ("structure" or "content"), the RULES' score with the detail they gave, and the JUDGE's score with its reason.
Decide, from the OUTPUT only, which one is right. Read the OUTPUT yourself; do not trust either explanation.

For every case answer:
- "ruling": "rules" if the RULES' score is the right one, "judge" if the JUDGE's is, "neither" if both are wrong.
- "correct": 0 to 1, the score the disputed dimension should have, by the scoring rule of the case.
- "explanation": at most 30 words, naming the evidence you saw in the OUTPUT.
- "rule_fix": empty if the RULES are right; otherwise one sentence on what the rule should do differently.

Answer with a JSON array only, no prose, no code fence:
[{"id": 1, "ruling": "rules", "correct": 0, "explanation": "...", "rule_fix": ""}, ...] with one object per case."""


@dataclass
class Settings:
    provider: str
    model: str
    base_url: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def settings() -> Settings:
    provider = (config.get("PDF_BENCH_JUDGE_PROVIDER") or "anthropic").lower()
    if provider not in DEFAULTS:
        raise ValueError(f"PDF_BENCH_JUDGE_PROVIDER = {provider!r}: use one of {', '.join(DEFAULTS)}")
    return Settings(provider, config.get("PDF_BENCH_JUDGE_MODEL") or DEFAULTS[provider],
                    config.get("PDF_BENCH_JUDGE_BASE_URL") or "")


def missing_settings() -> str:
    """Why the judge cannot call its model here, or an empty string."""
    try:
        s = settings()
    except ValueError as exc:
        return str(exc)
    key = "ANTHROPIC_API_KEY" if s.provider == "anthropic" else "OPENAI_API_KEY"
    if s.provider == "openai" and s.base_url:
        return ""
    return "" if config.get(key) else f"{key} is not set (.env)"


# --- the call -----------------------------------------------------------------------------------------------------

def _without_temperature(send):
    """Ask with temperature 0; some models refuse that parameter, then ask again without it."""
    try:
        return send(temperature=0)
    except Exception as exc:  # noqa: BLE001 - only the "temperature is not accepted" refusal is retried
        if "temperature" not in str(exc).lower():
            raise
        return send()


def _call(s: Settings, system: str, user: str) -> tuple[str, dict]:
    """(answer text, token usage). An answer cut by the token limit raises: it would be judged as missing."""
    if s.provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=config.get("ANTHROPIC_API_KEY"))
        msg = _without_temperature(lambda **kw: client.messages.create(
            model=s.model, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}], **kw))
        if getattr(msg, "stop_reason", None) not in ("end_turn", "stop_sequence", None):
            raise RuntimeError(f"judge answer incomplete (stop_reason = {msg.stop_reason})")
        u = getattr(msg, "usage", None)
        return ("".join(b.text for b in msg.content if getattr(b, "type", "") == "text"),
                {"input_tokens": getattr(u, "input_tokens", 0) or 0, "output_tokens": getattr(u, "output_tokens", 0) or 0})
    import openai

    key = (config.get("OPENAI_API_KEY") if not s.base_url else "") or "EMPTY"  # the OpenAI key goes to OpenAI only
    client = openai.OpenAI(base_url=s.base_url or None, api_key=key)
    resp = _without_temperature(lambda **kw: client.chat.completions.create(
        model=s.model, max_completion_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw))
    if getattr(resp.choices[0], "finish_reason", None) not in ("stop", None):
        raise RuntimeError(f"judge answer incomplete (finish_reason = {resp.choices[0].finish_reason})")
    u = getattr(resp, "usage", None)
    return (resp.choices[0].message.content or "",
            {"input_tokens": getattr(u, "prompt_tokens", 0) or 0, "output_tokens": getattr(u, "completion_tokens", 0) or 0})


def output_text(pages: list[str]) -> str:
    return "\n\n".join(f"<<<<< page {i} >>>>>\n{p}" for i, p in enumerate(pages, 1))


def prompt(pages: list[str], asked: list[tuple[int, str, str]]) -> str:
    """The user message: the output once, then the questions (id, kind, text) with the rule for their kind."""
    qs = "\n\n".join(f"Question {i} ({kind})\nExpected: {spec}\nScoring rule: {questions.RUBRIC.get(kind, '')}"
                     for i, kind, spec in asked)
    return f"OUTPUT:\n{output_text(pages)}\n\nQUESTIONS:\n\n{qs}"


def parse(answer: str, ids: list[int]) -> dict[int, dict]:
    """The judge's JSON array as {id: verdict}. Unreadable or missing questions are simply absent."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
    start, end = text.find("["), text.rfind("]")
    try:
        items = json.loads(text[start: end + 1]) if start >= 0 and end > start else []
    except json.JSONDecodeError:
        items = []
    out = {}
    for it in items if isinstance(items, list) else []:
        try:
            i = int(it["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if i in ids:
            out[i] = {k: _unit(it.get(k)) for k in ("content", "structure")} | {"reason": str(it.get("reason", ""))[:300]}
    return out


def _unit(v) -> float | None:
    try:
        return None if v is None else min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return None


# --- cache ------------------------------------------------------------------------------------------------------------

def _key(s: Settings, system: str, user: str) -> str:
    blob = json.dumps([PROMPT_VERSION, s.provider, s.model, s.base_url, system, user], ensure_ascii=False).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _cached(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _save(cache_dir: Path, key: str, s: Settings, answer: str, usage: dict, seconds: float) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(
        {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "judge": s.label, "prompt_version": PROMPT_VERSION,
         "usage": usage, "seconds": round(seconds, 2), "answer": answer}, ensure_ascii=False), encoding="utf-8")


# --- the judgement of one answer --------------------------------------------------------------------------------------

COLUMNS = ["judge_content", "judge_structure", "judge_reason"]
ARBITER_COLUMNS = ["arb_ruling", "arb_correct", "arb_reason", "arb_fix"]


def judge_frame(scored: pd.DataFrame, pages: list[str], *, call: bool = True, force: bool = False,
                cache_dir: Path = DEFAULT_DIR, tally: dict | None = None, arbitrate: bool = False) -> pd.DataFrame:
    """``scored`` (rows with a ``spec``) plus the judge's verdict for each row.

    ``call=False`` reads the cache only (nothing is sent anywhere): rows never judged stay empty.
    ``tally`` collects ``{"calls", "cached", "input_tokens", "output_tokens", "errors"}`` for the caller.
    ``arbitrate``: after the blind pass, the judge sees the rules' score for every question where they differ and
    says who is right (``arb_*`` columns): see :func:`arbitrate_frame`.
    """
    s = settings()
    tally = tally if tally is not None else {}
    for k in ("calls", "cached", "input_tokens", "output_tokens", "errors"):
        tally.setdefault(k, 0)
    out = scored.copy()
    out["judge_content"], out["judge_structure"], out["judge_reason"] = np.nan, np.nan, ""
    out["arb_ruling"], out["arb_correct"], out["arb_reason"], out["arb_fix"] = "", np.nan, "", ""
    asked = [(int(i) + 1, r.kind, r.spec) for i, r in enumerate(scored.itertuples()) if r.spec]
    for start in range(0, len(asked), CHUNK):
        group = asked[start: start + CHUNK]
        user = prompt(pages, group)
        key = _key(s, SYSTEM, user)
        hit = None if force else _cached(cache_dir, key)
        if hit:
            tally["cached"] += 1
        elif call:
            t0 = time.time()
            try:
                answer, usage = _call(s, SYSTEM, user)
            except Exception as exc:  # noqa: BLE001 - one failed call must not lose the others
                tally["errors"] += 1
                tally.setdefault("last_error", f"{type(exc).__name__}: {str(exc)[:400]}")
                for i, _, _ in group:
                    out.iloc[i - 1, out.columns.get_loc("judge_reason")] = f"judge error: {str(exc)[:120]}"
                continue
            _save(cache_dir, key, s, answer, usage, time.time() - t0)
            hit = {"answer": answer, "usage": usage}
            tally["calls"] += 1
            tally["input_tokens"] += usage["input_tokens"]
            tally["output_tokens"] += usage["output_tokens"]
        if not hit:
            continue
        verdicts = parse(hit["answer"], [i for i, _, _ in group])
        for i, _, _ in group:
            v = verdicts.get(i)
            row = i - 1
            if v is None:
                out.iloc[row, out.columns.get_loc("judge_reason")] = "judge gave no usable answer for this question"
                continue
            out.iloc[row, out.columns.get_loc("judge_content")] = np.nan if v["content"] is None else v["content"]
            out.iloc[row, out.columns.get_loc("judge_structure")] = np.nan if v["structure"] is None else v["structure"]
            out.iloc[row, out.columns.get_loc("judge_reason")] = v["reason"]
    return arbitrate_frame(out, pages, call=call, force=force, cache_dir=cache_dir, tally=tally) if arbitrate else out


# --- the judge compares its evaluation with the rules' ----------------------------------------------------------------

def disputed(row) -> str:
    """The dimension on which the rules and the judge differ by at least TOLERANCE ("" if they agree)."""
    for dim in ("structure", "content"):
        a, b = row.get(dim), row.get(f"judge_{dim}")
        if pd.notna(a) and pd.notna(b) and abs(a - b) >= TOLERANCE:
            return dim
    return ""


def _case(i: int, row, dim: str) -> str:
    rules = f"{row[dim]:.2f}" + (f" ({row['detail']})" if row["detail"] else "")
    return (f"Case {i} ({row['kind']}), disputed: {dim}\nExpected: {row['spec']}\nScoring rule: {questions.RUBRIC.get(row['kind'], '')}\n"
            f"RULES: {dim} = {rules}\nJUDGE: {dim} = {row[f'judge_{dim}']:.2f} ({row['judge_reason']})")


def arbitrate_frame(f: pd.DataFrame, pages: list[str], *, call: bool = True, force: bool = False,
                    cache_dir: Path = DEFAULT_DIR, tally: dict | None = None) -> pd.DataFrame:
    """For each question where the rules and the judge differ, the judge now sees both scores and both explanations
    and rules between them (``arb_ruling`` = rules / judge / neither, ``arb_correct`` the score it holds right,
    ``arb_reason``, and ``arb_fix`` for a rule that needs changing). The first pass stays blind, so this one cannot bias it."""
    s = settings()
    tally = tally if tally is not None else {}
    for k in ("calls", "cached", "input_tokens", "output_tokens", "errors"):
        tally.setdefault(k, 0)
    out = f.copy()
    for c, empty in (("arb_ruling", ""), ("arb_correct", np.nan), ("arb_reason", ""), ("arb_fix", "")):
        if c not in out:
            out[c] = empty
    cases = [(p, row, dim) for p, (_, row) in enumerate(out.iterrows()) if row["spec"] and (dim := disputed(row))]
    for start in range(0, len(cases), ARBITER_CHUNK):
        group = cases[start: start + ARBITER_CHUNK]
        numbered = [(n + 1, p, row, dim) for n, (p, row, dim) in enumerate(group)]
        user = f"OUTPUT:\n{output_text(pages)}\n\nCASES:\n\n" + "\n\n".join(_case(n, row, dim) for n, _, row, dim in numbered)
        key = _key(s, ARBITER + ARBITER_VERSION, user)
        hit = None if force else _cached(cache_dir, key)
        if hit:
            tally["cached"] += 1
        elif call:
            t0 = time.time()
            try:
                answer, usage = _call(s, ARBITER, user)
            except Exception as exc:  # noqa: BLE001 - one failed call must not lose the others
                tally["errors"] += 1
                tally.setdefault("last_error", f"{type(exc).__name__}: {str(exc)[:400]}")
                for _, p, _, _ in numbered:
                    out.iloc[p, out.columns.get_loc("arb_reason")] = f"arbitration error: {str(exc)[:120]}"
                continue
            _save(cache_dir, key, s, answer, usage, time.time() - t0)
            hit = {"answer": answer, "usage": usage}
            tally["calls"] += 1
            tally["input_tokens"] += usage["input_tokens"]
            tally["output_tokens"] += usage["output_tokens"]
        if not hit:
            continue
        for n, p, _, _ in numbered:
            v = _ruling(hit["answer"], n)
            if v is None:
                out.iloc[p, out.columns.get_loc("arb_reason")] = "no usable ruling for this case"
                continue
            for col, val in zip(ARBITER_COLUMNS, (v["ruling"], v["correct"], v["explanation"], v["rule_fix"])):
                out.iloc[p, out.columns.get_loc(col)] = val
    return out


def _ruling(answer: str, n: int) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())
    start, end = text.find("["), text.rfind("]")
    try:
        items = json.loads(text[start: end + 1]) if start >= 0 and end > start else []
    except json.JSONDecodeError:
        return None
    for it in items if isinstance(items, list) else []:
        try:
            if int(it["id"]) != n:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        ruling = str(it.get("ruling", "")).lower()
        if ruling not in ("rules", "judge", "neither"):
            return None
        return {"ruling": ruling, "correct": np.nan if _unit(it.get("correct")) is None else _unit(it.get("correct")),
                "explanation": str(it.get("explanation", ""))[:400], "rule_fix": str(it.get("rule_fix", ""))[:400]}
    return None


def rulings(f: pd.DataFrame) -> pd.DataFrame:
    """Per kind: how many disagreements, and who the judge held right after reading both sides."""
    g = f[f["arb_ruling"].ne("")] if "arb_ruling" in f else f.iloc[0:0]
    if g.empty:
        return pd.DataFrame(columns=["disagreements", "rules right", "judge right", "neither"])
    return g.groupby("kind").agg(disagreements=("kind", "size"), **{
        "rules right": ("arb_ruling", lambda x: int((x == "rules").sum())),
        "judge right": ("arb_ruling", lambda x: int((x == "judge").sum())),
        "neither": ("arb_ruling", lambda x: int((x == "neither").sum()))})


# --- comparing the two methods ----------------------------------------------------------------------------------------

TOLERANCE = 0.25  # a gap below this counts as agreement (the judge rounds fractions of cells or items)


def agreement(f: pd.DataFrame) -> pd.DataFrame:
    """Per kind: how often the rules and the judge agree on structure, and their mean scores."""
    g = f.dropna(subset=["structure", "judge_structure"]).copy()
    if g.empty:
        return pd.DataFrame(columns=["questions", "rules", "judge", "agree %", "rules stricter", "judge stricter"])
    g["gap"] = g["structure"] - g["judge_structure"]
    return g.groupby("kind").agg(
        questions=("kind", "size"), rules=("structure", "mean"), judge=("judge_structure", "mean"),
        **{"agree %": ("gap", lambda x: 100 * float((x.abs() < TOLERANCE).mean())),
           "rules stricter": ("gap", lambda x: int((x <= -TOLERANCE).sum())),
           "judge stricter": ("gap", lambda x: int((x >= TOLERANCE).sum()))}).round(2)


def disagreements(f: pd.DataFrame) -> pd.DataFrame:
    """The questions where the rules and the judge differ: the ones to read to decide who is right."""
    g = f.assign(disputed=f.apply(disputed, axis=1)) if len(f) else f.assign(disputed="")
    g = g[g["disputed"] != ""]
    cols = ("model", "pdf", "kind", "disputed", "expected", "content", "judge_content", "structure", "judge_structure", "detail",
            "judge_reason", "arb_ruling", "arb_correct", "arb_reason", "arb_fix", "question", "evidence")
    return g[[c for c in cols if c in g]]


def cohen_kappa(f: pd.DataFrame) -> float | None:
    """Agreement on 'recovered / not recovered' (structure >= 0.5) beyond chance; None when it is undefined."""
    g = f.dropna(subset=["structure", "judge_structure"])
    if len(g) < 5:
        return None
    a, b = (g["structure"] >= 0.5).to_numpy(), (g["judge_structure"] >= 0.5).to_numpy()
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return None if pe >= 1 else round((po - pe) / (1 - pe), 3)
