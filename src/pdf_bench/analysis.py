"""One analysis file per benchmark run, written so the scores and the extractions can be criticised afterwards.

``analysis/<dataset path>/analysis.md`` is the readable report (setup, scores, automatic checks, then per PDF and
model every element that was not fully recovered with the evidence found in the output); ``analysis.json`` holds
everything (ground truth, full outputs, recovered elements, every scored row) for a closer look.
Only answers already on disk are read: nothing is run here.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from . import judge, postprocess, recommend, results, view
from .convert import to_elements
from .elements import load_gt
from .parsers import lookup
from .score import norm
from .selection import Selection

DEFAULT_DIR = Path("analysis")
SNIPPET = 70  # characters of context on each side of the evidence
OUTPUT_EXCERPT = 2500  # characters of each output quoted in the Markdown (the JSON has all of it)
SUMMARY = {"table", "list", "reading_order", "numbers", "section_path", "page"}  # their `expected` is a description ("7 items"), not text to look for
RISKY = {"checkbox", "link", "footnote", "field", "figure", "caption", "header", "footer"}  # kinds judged on a heuristic


def evidence(expected: str, output: str) -> dict:
    """Where the expected text is in the output (best fuzzy alignment): the snippet and its similarity (0-100)."""
    text = re.sub(r"^H\d ", "", expected.lstrip("☒☐☑ ").strip())  # score rows write headings as "H2 title"
    text = text.replace(": (blank)", "")
    needle = norm(text)[:60]
    hay = norm(output)
    if not needle or not hay:
        return {"similarity": 0, "snippet": ""}
    m = fuzz.partial_ratio_alignment(needle, hay)
    a, b = max(0, m.dest_start - SNIPPET), min(len(hay), m.dest_end + SNIPPET)
    return {"similarity": round(m.score), "snippet": hay[a:b]}


def collect(selection: Selection, cache_dir: Path = results.DEFAULT_DIR, judge_mode: str = "off") -> dict:
    """Read the saved answers of the selection and score them. Missing answers are listed, not run.

    ``judge_mode``: ``off``; ``cached`` adds the LLM judge's saved verdicts (no call); ``run`` also asks the judge
    for the questions it has not answered yet (costs tokens).
    """
    docs = Path(selection.docs_dir)
    pdfs = [p for p in sorted(docs.glob("*.pdf")) if p.with_suffix(".gt.json").is_file()]
    if selection.docs_selected:
        pdfs = [p for p in pdfs if p.name in selection.docs_selected]
    entries = selection.entries()
    data: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": str(results.dataset_dir(docs / "x.pdf")),
                  "postprocess": list(selection.postprocess), "models": [], "ground_truth": {}, "runs": [], "missing": [],
                  "judge": judge.settings().label if judge_mode != "off" else None, "judge_usage": {}}
    for p in pdfs:
        gt = load_gt(p.with_suffix(".gt.json"))
        data["ground_truth"][p.name] = {"category": gt.category, "tested": gt.tested, "meta": gt.meta,
                                        "elements": [e.to_dict() for e in gt.elements]}
    for e in entries:
        parser = lookup(e.parser)
        with results.settings(parser, e.options or None):
            options = {k: parser.option(k) for k in sorted(parser.options)}
        data["models"].append({"label": e.label, "parser": e.parser, "version": parser.version, "options": options,
                               "not_applicable": view.not_applicable(e.parser)})
        for p in pdfs:
            a = results.load_cached(e.parser, p, cache_dir, e.options or None, e.label)
            if a is None:
                data["missing"].append({"model": e.label, "pdf": p.name})
                continue
            run = {"model": e.label, "pdf": p.name, "seconds": a["seconds"], "error": a["error"], "created": a["created"],
                   "pages": len(a["pages"]), "rows": [], "output": "\n\n".join(a["pages"]), "recovered": []}
            if not a["error"]:
                steps = list(selection.postprocess)
                scored = postprocess.apply(list(a["pages"]), steps) if steps else list(a["pages"])
                if steps:
                    run["output_postprocessed"] = "\n\n".join(scored)
                run["recovered"] = [x.to_dict() for x in to_elements(scored)]
                text = "\n\n".join(scored)  # what was scored: the evidence is looked for there
                scored_rows = view.frame(e.label, e.parser, p, list(a["pages"]), steps)
                if judge_mode != "off":
                    scored_rows = view.judged(scored_rows, e.parser, list(a["pages"]), steps, call=judge_mode == "run",
                                              tally=data["judge_usage"], arbitrate=True)
                for r in scored_rows.itertuples():
                    row = {"kind": r.kind, "expected": r.expected, "content": r.content,
                           "structure": None if pd.isna(r.structure) else r.structure, "detail": r.detail,
                           "evidence": evidence(r.expected, text) if r.content is not None and r.kind not in SUMMARY else None}
                    if judge_mode != "off":
                        row |= {"question": r.spec, "judge_content": None if pd.isna(r.judge_content) else r.judge_content,
                                "judge_structure": None if pd.isna(r.judge_structure) else r.judge_structure,
                                "judge_reason": r.judge_reason, "arb_ruling": r.arb_ruling,
                                "arb_correct": None if pd.isna(r.arb_correct) else r.arb_correct,
                                "arb_reason": r.arb_reason, "arb_fix": r.arb_fix}
                    run["rows"].append(row)
            data["runs"].append(run)
    data["flags"] = flags(data)
    return data


def frame(data: dict) -> pd.DataFrame:
    keys = ("kind", "expected", "content", "structure", "detail", "judge_content", "judge_structure", "judge_reason", "question",
            "arb_ruling", "arb_correct", "arb_reason", "arb_fix")
    rows = [{"model": r["model"], "pdf": r["pdf"], **{k: x.get(k) for k in keys}, "evidence": x["evidence"]}
            for r in data["runs"] for x in r["rows"]]
    f = pd.DataFrame(rows, columns=["model", "pdf", *keys, "evidence"])
    for c in ("content", "structure", "judge_content", "judge_structure", "arb_correct"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f[["judge_reason", "question", "arb_ruling", "arb_reason", "arb_fix"]] = f[
        ["judge_reason", "question", "arb_ruling", "arb_reason", "arb_fix"]].fillna("")
    return f


def flags(data: dict) -> list[str]:
    """Automatic doubts about the scores themselves (not about the models). Each one is worth a look."""
    out: list[str] = []
    f = frame(data)
    if f.empty:
        return out
    for (model, kind), g in f.groupby(["model", "kind"]):
        c, s = g["content"].dropna(), g["structure"].dropna()
        if len(s) and len(c) and c.mean() >= 0.99 and s.mean() <= 0.01 and len(g) >= 2:
            out.append(f"{model} / {kind}: the text is found everywhere (content 1.0) but the structure is never recovered (0.0): "
                       "check the model really does not mark it, rather than the scorer missing the marking")
    # a header or footer scores 1.0 when it is GONE from the output: a weak match is then what is expected
    weak = f[(f["structure"] == 1) & ~f["kind"].isin(("header", "footer")) & f["evidence"].map(lambda e: bool(e) and e["similarity"] < 90)]
    for _, r in weak.iterrows():
        out.append(f"{r.model} / {r.pdf} / {r.kind} '{r.expected[:40]}': structure 1.0 although the best match in the output "
                   f"is only {r.evidence['similarity']}% similar")
    odd = f[(f["content"] == 0) & (f["structure"] == 1)]
    for _, r in odd.iterrows():
        out.append(f"{r.model} / {r.pdf} / {r.kind} '{r.expected[:40]}': structure 1.0 with content 0.0 (contradiction)")
    by_pdf: dict[str, dict[str, str]] = {}
    for r in data["runs"]:
        by_pdf.setdefault(r["pdf"], {})[r["model"]] = r["output"]
    for pdf, outs in by_pdf.items():
        seen: dict[str, str] = {}
        for model, text in outs.items():
            if text.strip() and text in seen:
                out.append(f"{pdf}: {model} and {seen[text]} produced exactly the same output")
            seen.setdefault(text, model)
    for r in data["runs"]:
        if r["error"]:
            out.append(f"{r['model']} / {r['pdf']}: failed ({str(r['error'])[:120]})")
        elif not r["output"].strip():
            out.append(f"{r['model']} / {r['pdf']}: empty output")
    for m in data["missing"]:
        out.append(f"{m['model']} / {m['pdf']}: no saved answer (not run)")
    for m in data["models"]:
        if m["label"] in recommend.REFERENCE_MODELS:
            out.append(f"{m['label']} is the reference: it wrote these PDFs, so its scores are a ceiling to compare with, not a "
                       "result. It is left out of the recommendation.")
    return out


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = ["" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r]
        lines.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def _judge_section(f: pd.DataFrame, data: dict) -> list[str]:
    done = f.dropna(subset=["structure", "judge_structure"])
    o = ["## Rules against the LLM judge", "",
         f"Judge: `{data['judge']}`. Each expected element is a question; the rules and the judge answer it independently "
         f"(`judge_*` columns). {len(done)} structure questions answered by both. Agreement = gap below {judge.TOLERANCE}; "
         "kappa is on recovered / not recovered (structure >= 0.5).", ""]
    per = []
    for model, g in done.groupby("model", sort=False):
        per.append({"model": model, "questions": len(g), "rules": g.groupby("kind")["structure"].mean().mean(),
                    "judge": g.groupby("kind")["judge_structure"].mean().mean(),
                    "agree %": 100 * float(((g["structure"] - g["judge_structure"]).abs() < judge.TOLERANCE).mean()),
                    "kappa": judge.cohen_kappa(g)})
    o += [_md_table(pd.DataFrame(per)), "### By kind of element", "", _md_table(judge.agreement(done).reset_index()),
          "### Where they disagree", ""]
    diff = judge.disagreements(f)
    o += [_md_table(diff.assign(expected=diff["expected"].str[:60])) if len(diff) else "_They agree on every question._", ""]
    return o


def markdown(data: dict) -> str:
    f = frame(data)
    o = [f"# Benchmark analysis: {data['dataset']}", "", f"Generated {data['generated']}. {len(data['ground_truth'])} PDF(s), "
         f"{len(data['models'])} model(s), {len(data['runs'])} answer(s) read, {len(data['missing'])} missing. "
         + (f"Post-processing applied before scoring: {', '.join(data['postprocess'])}." if data["postprocess"] else "No post-processing."),
         "Structure marked n/a: the tool's documented output cannot express that kind (see the Models table).", "",
         "Scores: **content** = the text is found in the output (fuzzy, >= 90%); **structure** = recovered as the right kind of "
         "element in the right place. `evidence` is the best match of the expected text in the output, with its similarity. "
         "Kinds judged by a heuristic (" + ", ".join(sorted(RISKY)) + ") are listed again under \"Recovered by a heuristic "
         "check\" when they scored 1.0, so the evidence can be read. Tables and lists have no evidence (the row describes "
         "them, e.g. '7 items'): compare the Output block with the ground truth in analysis.json.", "",
         "## Models", ""]
    o.append(_md_table(pd.DataFrame([{"model": m["label"], "parser": m["parser"], "version": m["version"],
                                      "options": json.dumps(m["options"], ensure_ascii=False),
                                      "not applicable": ", ".join(m["not_applicable"])} for m in data["models"]])))
    o += ["## Scores", ""]
    if not f.empty:
        per = []
        for model, g in f.groupby("model", sort=False):
            k = g.groupby("kind")[["content", "structure"]].mean()
            runs = [r for r in data["runs"] if r["model"] == model]
            per.append({"model": model, "PDFs": len(runs), "content": k["content"].mean(), "structure": k["structure"].mean(),
                        "seconds": round(sum(r["seconds"] or 0 for r in runs), 1), "errors": sum(1 for r in runs if r["error"])})
        o += [_md_table(pd.DataFrame(per)), "### Structure by kind of element", ""]
        o.append(_md_table(f.pivot_table(index="kind", columns="model", values="structure", aggfunc="mean").reset_index()))
        o += ["### Content by kind of element", ""]
        o.append(_md_table(f.pivot_table(index="kind", columns="model", values="content", aggfunc="mean").reset_index()))
    if data.get("judge") and not f.empty and f["judge_structure"].notna().any():
        o += _judge_section(f, data)
    o += ["## Automatic checks on the scores", ""]
    o += [f"* {x}" for x in data["flags"]] or ["_Nothing flagged._"]
    o += ["", "## Per PDF", ""]
    for pdf, gt in data["ground_truth"].items():
        kinds: dict[str, int] = {}
        stack = list(gt["elements"])
        while stack:
            e = stack.pop()
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
            stack += e.get("children", [])
        o += [f"### {pdf} ({gt['category']})", "", "Expected: " + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())), ""]
        for r in (x for x in data["runs"] if x["pdf"] == pdf):
            head = f"#### {pdf} / {r['model']}  ·  {r['seconds']} s"
            if r["error"]:
                o += [head, "", f"Failed: {r['error']}", ""]
                continue
            g = pd.DataFrame(r["rows"])
            bad = g[(g["structure"].fillna(1) < 1) | (g["content"].fillna(1) < 1)] if not g.empty else g
            risky = g[g["kind"].isin(RISKY) & (g["structure"] == 1)] if not g.empty else g
            o += [head, ""]
            if bad.empty:
                o.append("Every expected element recovered.\n")
            else:
                t = bad.assign(evidence=bad["evidence"].map(lambda e: f"{e['similarity']}%: {e['snippet']}" if e else ""))
                cols = ["kind", "expected", "content", "structure", "detail", "evidence"]
                if data.get("judge"):
                    cols += ["judge_content", "judge_structure", "judge_reason"]
                o.append(_md_table(t[cols]))
            if not risky.empty:
                t = risky.assign(evidence=risky["evidence"].map(lambda e: f"{e['similarity']}%: {e['snippet']}" if e else ""))
                o += ["Recovered by a heuristic check (to confirm):", "", _md_table(t[["kind", "expected", "structure", "evidence"]])]
            text = r["output"]
            cut = text[:OUTPUT_EXCERPT] + (f"\n[... {len(text) - OUTPUT_EXCERPT} more characters in analysis.json]" if len(text) > OUTPUT_EXCERPT else "")
            o += ["<details><summary>Output</summary>", "", "```", cut, "```", "", "</details>", ""]
    return "\n".join(o)


def write(selection: Selection, out_dir: Path = DEFAULT_DIR, cache_dir: Path = results.DEFAULT_DIR,
          judge_mode: str = "off") -> tuple[Path, Path]:
    data = collect(selection, cache_dir, judge_mode)
    target = Path(out_dir) / data["dataset"]
    target.mkdir(parents=True, exist_ok=True)
    md, js = target / "analysis.md", target / "analysis.json"
    md.write_text(markdown(data), encoding="utf-8")
    js.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return md, js


# --- the judge's own report ---------------------------------------------------------------------------------------

JUDGE_QUESTION_CHARS = 700  # characters of a question quoted in the report (the JSON has all of it)


def judge_report(data: dict) -> str:
    """Markdown report on rules against judge: coverage, agreement, then every disagreement with both explanations."""
    f = frame(data)
    ask = f[f["question"] != ""]
    done = ask.dropna(subset=["structure", "judge_structure"])
    both_c = ask.dropna(subset=["content", "judge_content"])
    o = [f"# Judge analysis: {data['dataset']}", "",
         f"Generated {data['generated']}. Judge: `{data['judge']}` (prompt version {judge.PROMPT_VERSION}). {len(data['models'])} model(s), "
         f"{len(data['runs'])} answer(s), post-processing: {', '.join(data['postprocess']) or 'none'}.",
         "", "Each expected element is a question. The rules (deterministic scorer) and the LLM judge answered it independently from the same "
         "output. A disagreement means a rule is too strict or too lax, or the judge is wrong: the reasons below say which. "
         "Agreement = gap below " + str(judge.TOLERANCE) + "; kappa is on recovered / not recovered (>= 0.5).", "",
         "## Coverage", ""]
    unanswered = ask[ask["structure"].notna() & ask["judge_structure"].isna() & ask["judge_reason"].ne("")]
    o += [f"* questions asked: {len(ask)}; structure judged and scored by the rules: {len(done)}; content judged: {len(both_c)}",
          f"* not judged yet (run the judge): {int((ask['structure'].notna() & ask['judge_structure'].isna() & ask['judge_reason'].eq('')).sum())}",
          f"* judge errors or unusable answers: {len(unanswered)}" + (f", e.g. {unanswered['judge_reason'].iloc[0]!r}" if len(unanswered) else ""),
          f"* judge usage in the last run that wrote this file: {data.get('judge_usage') or 'none (verdicts read from the saved ones)'}", ""]
    if done.empty:
        return "\n".join(o + ["_No question has both a rules score and a judge verdict yet._", ""])
    o += ["## Per model", ""]
    per = []
    for model, g in done.groupby("model", sort=False):
        gap = g["structure"] - g["judge_structure"]
        c = both_c[both_c["model"] == model]
        per.append({"model": model, "questions": len(g), "rules": g.groupby("kind")["structure"].mean().mean(),
                    "judge": g.groupby("kind")["judge_structure"].mean().mean(), "mean gap (rules - judge)": gap.mean(),
                    "agree %": 100 * float((gap.abs() < judge.TOLERANCE).mean()), "kappa": judge.cohen_kappa(g),
                    "content agree %": 100 * float(((c["content"] - c["judge_content"]).abs() < judge.TOLERANCE).mean()) if len(c) else None})
    o += [_md_table(pd.DataFrame(per)),
          "A negative mean gap means the judge is more generous than the rules, a positive one that the rules are more generous.", "",
          "## Per kind of element (all models)", "", _md_table(judge.agreement(done).reset_index()), "",
          "## Per PDF", ""]
    pp = done.assign(agree=((done["structure"] - done["judge_structure"]).abs() < judge.TOLERANCE)).groupby("pdf").agg(
        questions=("agree", "size"), **{"agree %": ("agree", lambda x: 100 * float(x.mean()))}).round(1)
    o += [_md_table(pp.reset_index()), "", "## Disagreements", ""]
    diff = judge.disagreements(ask)
    o += [f"{len(diff)} question(s) where the rules and the judge differ by {judge.TOLERANCE} or more on structure or on content. "
          "After its blind pass the judge was shown both scores and both explanations and ruled on each (*ruling*: who is right, "
          "after reading the output again).", ""]
    judged = diff[diff["arb_ruling"] != ""]
    if len(diff) and judged.empty:
        o += ["_The judge has not ruled on them yet: press *Run the judge* with *the judge rules on its disagreements* ticked._", ""]
    elif len(judged):
        r = judge.rulings(diff).reset_index()
        total = {"kind": "all", **{c: int(r[c].sum()) for c in r.columns[1:]}}
        o += ["### Who is right, according to the judge", "", _md_table(pd.concat([r, pd.DataFrame([total])])),
              "`rules right` = the judge confirms the rules and withdraws its own score; `judge right` = the rules should change (see *rule fix*).", ""]
    if diff.empty:
        o.append("_The rules and the judge agree on every question._")
    for kind, g in diff.groupby("kind"):
        o += [f"### {kind}: {len(g)} disagreement(s)", ""]
        for r in g.itertuples():
            dim = r.disputed
            rules_v, judge_v = (r.structure, r.judge_structure) if dim == "structure" else (r.content, r.judge_content)
            ev = (f" Best match in the output: {r.evidence['similarity']}%: {r.evidence['snippet']!r}"
                  if isinstance(r.evidence, dict) and r.evidence else "")
            o += [f"* **{r.model} / {r.pdf}** — {dim}: rules {rules_v:.2f} ({r.detail or 'no detail'}) · judge {judge_v:.2f} ({r.judge_reason})"]
            if r.arb_ruling:
                o += [f"  * **ruling: {"the rules are" if r.arb_ruling == "rules" else "the judge is" if r.arb_ruling == "judge" else "neither is"} right** (should be {r.arb_correct:.2f}): {r.arb_reason}"
                      + (f" — rule fix: {r.arb_fix}" if r.arb_fix else "")]
            elif r.arb_reason:
                o += [f"  * ruling: {r.arb_reason}"]
            o += [f"  * expected: {str(r.question)[:JUDGE_QUESTION_CHARS].replace(chr(10), ' / ')}" + ev]
        o.append("")
    o += ["## How to use this file", "",
          "Give it to Claude with the question: do you agree with the judge's rulings, and which rules of `score.py` to change "
          "(start with the cases ruled `judge` or `neither`). Then edit the rule, press *Rerun evaluation* on the Compare page and run the report again: "
          "nothing is parsed or judged again unless the output or the question changed.", ""]
    return "\n".join(o)


def write_judge(selection: Selection, out_dir: Path = DEFAULT_DIR, cache_dir: Path = results.DEFAULT_DIR,
                judge_mode: str = "cached") -> tuple[Path, Path]:
    """``judge_analysis.md`` and ``.json`` next to the analysis files. ``judge_mode`` = ``cached`` reads the saved verdicts."""
    data = collect(selection, cache_dir, judge_mode)
    target = Path(out_dir) / data["dataset"]
    target.mkdir(parents=True, exist_ok=True)
    md, js = target / "judge_analysis.md", target / "judge_analysis.json"
    md.write_text(judge_report(data), encoding="utf-8")
    js.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return md, js
