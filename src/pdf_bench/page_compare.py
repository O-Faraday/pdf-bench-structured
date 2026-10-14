"""Compare page: run the chosen models on every PDF of a folder, read the scores, then open any (PDF, model)
pair to compare the ground truth with what the model extracted.

Layout: a header that says which evaluation produced the numbers, a collapsible *Selection and run* panel, then
tabs: *Ranking* (raw -> after post-processing, recommendation, export), *By kind* (heatmaps with the number of
questions), *Detail* (one PDF, one model) and *About the scores*.

Answers are cached in ``results/`` (see :mod:`pdf_bench.results`), so a second visit costs nothing.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import pandas as pd  # noqa: E402
import pymupdf  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402

import pdf_bench  # noqa: E402
from pdf_bench import config, postprocess, pricing, recommend, rerun, results, selection, view  # noqa: E402
from pdf_bench.convert import to_elements  # noqa: E402
from pdf_bench.elements import load_gt  # noqa: E402
from pdf_bench.parsers import REFERENCES, REGISTRY, lookup  # noqa: E402

config.load_env()
st.title("Compare models with the ground truth")

saved = selection.load(st.session_state.get("selection_file", str(selection.DEFAULT_PATH)))

BAR = dict(min_value=0, max_value=1, format="%.2f")
REF = recommend.REFERENCE_MODELS


def bars(columns) -> dict:
    return {c: st.column_config.ProgressColumn(c, **BAR) for c in columns}


def tag(name: str) -> str:
    """How a model is named in tables and charts: the reference (it wrote the PDFs) carries a star."""
    return f"{name} ★" if name in REF else name


def evaluation_rev() -> str:
    """Short fingerprint of the evaluation code (scorer, converter, post-processing...): same fingerprint, same rules."""
    h = hashlib.sha1()
    root = Path(pdf_bench.__file__).parent
    for name in rerun.EVALUATION:
        if (f := root / f"{name}.py").is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:8]


# --- selection and run --------------------------------------------------------------------------------------------

docs_dir = Path(saved.docs_dir)
every = [p for p in sorted(docs_dir.glob("*.pdf")) if p.with_suffix(".gt.json").is_file()] if docs_dir.is_dir() else []
pdfs = [p for p in every if p.name in saved.docs_selected] if saved.docs_selected else every
if not pdfs:
    st.info("No PDF with a ground truth (`*.gt.json`) selected: create or select some on the *Documents and models* page.")
    st.stop()
saved.apply()  # settings saved with the selection (model ids, URLs, volume...) count for readiness too
entries = {e.label: e for e in saved.entries()}


def why_not(e) -> str:
    if e.parser not in REGISTRY and e.parser not in REFERENCES:
        return "unknown parser"
    with results.settings(lookup(e.parser), e.options):
        return lookup(e.parser).status()


ready = {l: e for l, e in entries.items() if not why_not(e)}
models = list(ready)
if not models:
    st.info("No model selected: choose and save some on the *Documents and models* page.")
    st.stop()

store: dict = st.session_state.setdefault("answers", {})
for n in models:  # answers saved by an earlier run are picked up without calling the model again
    for p in pdfs:
        if (n, str(p)) not in store and (hit := results.load_cached(ready[n].parser, p, options=ready[n].options or None, label=n)):
            store[(n, str(p))] = hit

with st.expander("Selection and run", expanded=not any((n, str(p)) in store for n in models for p in pdfs)):
    if skipped := [l for l in entries if l not in ready]:
        st.warning("Selected but not ready here, so skipped: " + ", ".join(f"{l} ({why_not(entries[l])})" for l in skipped))
    st.markdown("**Models**: " + ", ".join(
        f"`{l}`" + (f" (= {e.parser}, {', '.join(f'{k}={v}' for k, v in e.options.items())})" if e.label != e.parser else "")
        for l, e in ready.items()))
    st.markdown(f"**PDFs** (`{docs_dir}`): {len(pdfs)} of {len(every)}: " + ", ".join(p.name for p in pdfs))
    st.caption("Change the selection on the *Documents and models* page, then press *Save selection*.")
    hosted = sorted({ready[l].parser for l in models if lookup(ready[l].parser).hosted})
    ok_hosted = True
    if hosted:
        ok_hosted = st.checkbox(f"{', '.join(hosted)} send the PDFs to a third party and cost money: run them",
                                value=saved.allow_hosted)
    st.caption(f"{len(models)} model(s) × {len(pdfs)} PDF(s) = {len(models) * len(pdfs)} run(s)")
    c1, c2, c3, _ = st.columns([1, 1, 1, 2])
    force_models = c2.multiselect("Parse again only", models, help="Models whose saved answers are ignored: they are parsed again "
                                  "(and billed again). The others keep their saved answers and are not called.")
    run_now = c1.button("Run the selected models", type="primary", disabled=not ok_hosted)
    c3.button("Rerun evaluation", on_click=lambda: st.session_state.update(reloaded=rerun.reevaluate()),
              help="Score the saved answers again with the evaluation code as it is on disk (scorer, converter, "
                   "post-processing, questions): edit it, press this, no restart and no parsing.")
    if reloaded := st.session_state.pop("reloaded", None):
        st.success("Evaluation reloaded (" + ", ".join(reloaded) + "): the saved answers are scored again, nothing was parsed.")

if run_now:
    todo = [(n, p) for n in models for p in pdfs]
    bar = st.progress(0.0)
    for k, (n, p) in enumerate(todo):
        bar.progress(k / len(todo), text=f"{n} on {p.name} ({k + 1}/{len(todo)})")
        e = ready[n]
        store[(n, str(p))] = results.run(e.parser, p, force=n in force_models, options=e.options or None, label=n)
    bar.empty()

have = {(n, p): store[(n, str(p))] for n in models for p in pdfs if (n, str(p)) in store}
if not have:
    st.info("Press *Run the selected models*.")
    st.stop()


@st.cache_data(show_spinner="Scoring...")
def scored_frame(name: str, parser: str, pdf: str, created: str, pages: tuple, steps: tuple) -> pd.DataFrame:
    return view.frame(name, parser, pdf, list(pages), steps)


def frames_for(steps: tuple) -> dict:
    return {(n, p): scored_frame(n, ready[n].parser, str(p), a["created"], tuple(a["pages"]), steps)
            for (n, p), a in have.items() if not a["error"]}


steps = tuple(saved.postprocess)
errors = {k: a["error"] for k, a in have.items() if a["error"]}
frames = frames_for(steps)
raw_frames = frames_for(()) if steps else frames

# header: which evaluation produced these numbers
made = sorted(a["created"] for a in have.values())
st.caption(
    f"Evaluation `{evaluation_rev()}` · {len(models)} model(s) × {len(pdfs)} PDF(s) · post-processing: "
    + (", ".join(steps) if steps else "none")
    + f" · answers from {made[0][:10]} to {made[-1][:10]}"
    + (f" · ★ = reference: {', '.join(m for m in models if m in REF)} wrote these PDFs, shown but never ranked" if any(m in REF for m in models) else ""))
if missing := len(models) * len(pdfs) - len(have):
    st.warning(f"{missing} run(s) have no answer yet: open *Selection and run* and press *Run the selected models*.")

PASTEL = LinearSegmentedColormap.from_list("pastel_yellow_blue", ["#fff6b3", "#d9efb8", "#a9dcd0", "#8ec5e8"])


def heatmap(data: pd.DataFrame, title: str):
    """Seaborn heatmap, pastel yellow (low) to blue (high); rows = what is measured, columns = models.
    Empty cells (no such element) in grey."""
    data = data.astype(float)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.5 * len(data.columns) + 2.5), max(2.5, 0.42 * len(data) + 1.4)))
    sns.heatmap(data, vmin=0, vmax=1, cmap=PASTEL, annot=True, fmt=".2f", linewidths=0.5, linecolor="white",
                cbar_kws={"label": title, "shrink": 0.8}, mask=data.isna(), ax=ax, annot_kws={"size": 9})
    ax.set_facecolor("#e6e6e6")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="left")
    fig.tight_layout()
    return fig


def show_heatmap(data: pd.DataFrame, title: str, key: str) -> None:
    fig = heatmap(data, title)
    st.pyplot(fig, width="content", alt=f"Heatmap: {title}")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    b1, b2, _ = st.columns([1, 1, 4])
    b1.download_button("Download PNG", buf.getvalue(), f"{key}.png", "image/png", key=f"png_{key}")
    b2.download_button("Download CSV", data.to_csv().encode("utf-8"), f"{key}.csv", "text/csv", key=f"csv_{key}")


def not_applicable_kinds(label: str) -> list[str]:
    """Kinds of the selected PDFs that this model's tool cannot express (their structure is left out of the means)."""
    here = {k for (m, _), f in frames.items() if m == label for k in f["kind"].unique()}
    return [k for k in view.not_applicable(ready[label].parser) if k in here]


def pdf_pages(path: Path) -> int:
    with pymupdf.open(path) as d:
        return len(d)


def means(f: pd.DataFrame) -> dict:
    """Mean of the per-kind means, so a 200-cell table does not outweigh a single heading."""
    if f.empty:
        return {"content": None, "structure": None}
    k = f.groupby("kind")[["content", "structure"]].mean()
    return {c: (None if k[c].dropna().empty else round(float(k[c].dropna().mean()), 3)) for c in ("content", "structure")}


def pooled(fr: dict, model: str) -> pd.DataFrame:
    mine = [f.assign(pdf=p.name) for (m, p), f in fr.items() if m == model]
    return pd.concat(mine) if mine else pd.DataFrame()


# --- numbers shared by the tabs -----------------------------------------------------------------------------------

rows, summary = [], []
scan_pdfs = {p.name for p in pdfs if load_gt(docs_dir / f"{p.stem}.gt.json").category == "scan"}
for n in models:
    allf, rawf = pooled(frames, n), pooled(raw_frames, n)
    over, over_raw = means(allf), means(rawf)
    runs = [a for (m, _), a in have.items() if m == n]
    n_a = not_applicable_kinds(n)
    cost = pricing.estimate(ready[n].parser, lambda k, d="", e=ready[n]: e.options.get(k) or lookup(e.parser).option(k, d), runs,
                            sum(pdf_pages(p) for (m, p) in have if m == n))
    total = round(sum(a["seconds"] or 0 for a in runs), 1)
    n_err = sum(1 for a in runs if a["error"])
    gain = None if over["structure"] is None or over_raw["structure"] is None else round(over["structure"] - over_raw["structure"], 3)
    rows.append({"model": tag(n), "PDFs": f"{allf['pdf'].nunique() if len(allf) else 0}/{len(pdfs)}",
                 "structure": over["structure"], "content": over["content"],
                 **({"raw structure": over_raw["structure"], "post-processing gain": gain} if steps else {}),
                 "not applicable": ", ".join(n_a), "s / PDF": round(total / len(runs), 1) if runs else None,
                 "est. cost $": cost.usd, "cost basis": cost.basis, "errors": n_err})
    focus = recommend.focus_scores(allf, scan_pdfs)
    summary.append({"name": n, "free": cost.free, "cost": cost.usd, "seconds": total, "runs": allf["pdf"].nunique() if len(allf) else 0,
                    "expected_runs": len(pdfs), "errors": n_err, "not_applicable": n_a, **focus})

t_rank, t_kind, t_detail, t_about = st.tabs(["Ranking", "By kind", "Detail", "About the scores"])

# --- Ranking -------------------------------------------------------------------------------------------------------

with t_rank:
    table = pd.DataFrame(rows)
    table["_ref"] = table["model"].str.endswith("★")
    table = table.sort_values(["_ref", "structure"], ascending=[False, False], na_position="last").drop(columns="_ref").set_index("model")
    cols = ["structure", "content"] + (["raw structure"] if steps else [])
    st.dataframe(table, width="stretch", column_config={
        **bars(cols),
        "post-processing gain": st.column_config.NumberColumn("post-processing gain", format="%+.2f",
                                                              help="structure after post-processing minus structure of the raw output."),
        "not applicable": st.column_config.TextColumn("not applicable", help="Kinds this tool's documented output cannot express: left out of its structure mean, shown grey in the heatmaps."),
        "PDFs": st.column_config.TextColumn("PDFs", help="Documents with an answer / documents in this selection."),
        "s / PDF": st.column_config.NumberColumn("s / PDF", help="Mean time per PDF as recorded when it ran. Local tools and APIs are not comparable."),
        "est. cost $": st.column_config.NumberColumn("est. cost $", format="%.4f", help="Estimated from list prices (see the basis), or the amount billed when the service reports it. 0 = runs locally. Empty = no known rate."),
        "errors": st.column_config.NumberColumn("errors", help="Runs that failed: see below.")})
    st.caption("structure = recovered as the right element in the right place; content = the text is in the output. Each is the mean over "
               "element kinds (a 200-cell table does not outweigh one heading); kinds the tool cannot express are left out of its structure mean. "
               + ("*raw structure* is the model's output untouched; the gain is what the post-processing adds, the same rules for every model. "
                  if steps else "No post-processing is applied. ")
               + f"Costs are list prices read on {pricing.CHECKED}.")
    if errors:
        with st.expander(f"{len(errors)} failed run(s)"):
            st.dataframe(pd.DataFrame([{"model": n, "pdf": p.name, "error": e} for (n, p), e in errors.items()]),
                         hide_index=True, width="stretch")

    if steps:
        with st.expander("What each post-processing step adds, per model"):
            st.caption("Structure of each step alone minus the raw structure (the same answers, scored again). "
                       "The steps never read the ground truth.")
            if st.checkbox("Compute (scores every answer once per step)", key="per_step"):
                gains = {}
                for n in models:
                    base = means(pooled(raw_frames, n))["structure"]
                    row = {}
                    for s, (label, _) in postprocess.STEPS.items():
                        one = means(pooled(frames_for((s,)), n))["structure"]
                        row[label] = None if base is None or one is None else round(one - base, 3)
                    gains[tag(n)] = row
                g = pd.DataFrame(gains).T
                st.dataframe(g, width="stretch", column_config={c: st.column_config.NumberColumn(c, format="%+.2f") for c in g.columns})

    st.markdown("#### Recommendation")
    adv = recommend.advise(summary)
    st.caption("Ranked on the structure of tables, figures and captions, form fields and checkboxes, plus the text read from scans "
               "(OCR); the overall structure score breaks ties. Only models that ran on every selected PDF without error. "
               "Free = runs on your machine; paid = a service or hosted model.")
    rc1, rc2 = st.columns(2)
    for col, cat, title in ((rc1, "free", "Free"), (rc2, "paid", "Paid")):
        col.markdown(f"**{title}**")
        if adv[cat]:
            col.dataframe(pd.DataFrame([{"rank": m["rank"], "model": m["name"], "focus score": m["focus"],
                                         "overall structure": m["overall"], "why": m["why"]} for m in adv[cat]]).set_index("rank"),
                          width="stretch", column_config=bars(["focus score", "overall structure"]))
        else:
            col.caption("None in this selection.")
    for n, why in adv["excluded"]:
        st.caption(f"Not ranked: {n} ({why}).")
    for note in adv["notes"]:
        st.caption(note)
    st.caption("Based on the selected synthetic PDFs only: it tells which tool recovers these structures best, not which is best on your real documents.")

    st.markdown("#### Export")
    e1, e2, e3 = st.columns([1, 1, 2])
    e1.download_button("Ranking (CSV)", table.to_csv().encode("utf-8"), "ranking.csv", "text/csv")
    e2.download_button("Ranking (JSON)", table.reset_index().to_json(orient="records", indent=1).encode("utf-8"), "ranking.json", "application/json")
    if e3.button("Write the analysis files", help="analysis/<dataset path>/analysis.md and analysis.json: scores, elements not "
                                                  "recovered with their evidence, automatic checks on the scores, and the outputs."):
        from pdf_bench import analysis

        md_path, json_path = analysis.write(saved)
        st.success(f"Written: `{md_path}` (readable report) and `{json_path}` (everything: full outputs, recovered elements).")

# --- By kind --------------------------------------------------------------------------------------------------------

with t_kind:
    k1, k2 = st.columns([1, 1])
    metric = k1.radio("Metric", ["structure", "content"], horizontal=True)
    version = k2.radio("Output", ["after post-processing", "raw"], horizontal=True, disabled=not steps,
                       help="Without post-processing steps there is only one version.") if steps else "after post-processing"
    cur = raw_frames if version == "raw" else frames
    if metric == "content":
        st.caption("content only checks that the text is present in the output. A checkbox, link, footnote or figure caption "
                   "scores 1.0 as soon as its label is there, even if the tick, the link target or the footnote mark was "
                   "not recovered: use **structure** to judge those.")
    by_kind, by_pdf, n_kind = {}, {}, {}
    for n in models:
        allf = pooled(cur, n)
        if len(allf):
            by_kind[tag(n)] = allf.groupby("kind")[metric].mean().round(3)
            by_pdf[tag(n)] = allf.groupby("pdf")[metric].apply(lambda s: None if s.dropna().empty else round(float(s.mean()), 3))
            for kind, c in allf.groupby("kind")[metric].count().items():
                n_kind[kind] = max(n_kind.get(kind, 0), int(c))
    sub_kind, sub_pdf = st.tabs(["By kind of element", "By PDF"])
    with sub_kind:
        if by_kind:
            k = pd.DataFrame(by_kind)
            k.index = [f"{i}  (n={n_kind.get(i, 0)})" for i in k.index]
            show_heatmap(k, f"{metric} score", f"by_kind_{metric}_{'raw' if version == 'raw' else 'after'}")
            st.caption("Rows: kind of element and **n**, the number of questions behind each cell: a cell built on a handful of "
                       "questions is an anecdote, not a measure. Grey: no expected element of this kind, or the tool cannot express it.")
    with sub_pdf:
        if by_pdf:
            d = pd.DataFrame(by_pdf)
            d.index = [f"{i}  ({load_gt(docs_dir / f'{Path(i).stem}.gt.json').category})" for i in d.index]
            show_heatmap(d, f"{metric} score", f"by_pdf_{metric}_{'raw' if version == 'raw' else 'after'}")
            st.caption("Mean over the expected elements of each PDF; the document category is in brackets. Pick one in *Detail* to see why.")

# --- Detail: one PDF, one model --------------------------------------------------------------------------------------


def detail() -> None:
    d1, d2 = st.columns(2)
    pick = d1.selectbox("PDF", [p.name for p in pdfs])
    pdf = docs_dir / pick
    gt = load_gt(pdf.with_suffix(".gt.json"))
    names = [n for n in models if (n, pdf) in have]
    if not names:
        st.info("No model has an answer for this PDF yet.")
        return
    name = d2.selectbox("Model", names, format_func=tag)
    a = have[(name, pdf)]
    p = lookup(ready[name].parser)
    left, right = st.columns([1, 2])
    with pymupdf.open(pdf) as doc:
        page_no = left.number_input("Page", 1, len(doc), 1) if len(doc) > 1 else 1
        left.image(doc[page_no - 1].get_pixmap(dpi=75).tobytes("png"), caption=f"{pick}, page {page_no} of {len(doc)}")
    right.markdown(f"**Ground truth**: `{gt.category}`; it asserts " + ", ".join(f"`{t}`" for t in gt.tested))
    right.json(gt.counts())
    if gt.meta.get("text_layer") is False:
        right.caption("This PDF is a scan: it has no text layer, so a parser needs OCR.")
    right.caption(f"{p.label}  ·  version {p.version or '-'}  ·  options {json.dumps(a['options'])}  ·  {a['seconds']} s  ·  "
                  f"{'read from disk' if a['cached'] else 'run now'} ({a['created']})")
    if a["error"]:
        st.error(a["error"])
        return

    f = frames[(name, pdf)]
    over = means(f)
    m1, m2, m3 = st.columns(3)
    m1.metric("Content", "–" if over["content"] is None else f"{over['content']:.2f}")
    m2.metric("Structure", "–" if over["structure"] is None else f"{over['structure']:.2f}")
    m3.metric("Elements expected", len(f))

    tab_el, tab_out, tab_json = st.tabs(["Elements", "Output", "JSON"])
    with tab_el:
        st.dataframe(f.groupby("kind").agg(elements=("kind", "size"), content=("content", "mean"),
                                           structure=("structure", "mean")).round(3), width="stretch")
        only_bad = st.checkbox("Only the elements not fully recovered", value=True)
        detail_f = f.drop(columns=["spec"], errors="ignore")
        shown = detail_f[(detail_f["structure"].fillna(1) < 1) | (detail_f["content"].fillna(1) < 1)] if only_bad else detail_f
        st.dataframe(shown, width="stretch", hide_index=True, column_config={
            "kind": st.column_config.TextColumn("kind", help="Kind of element in the ground truth."),
            "expected": st.column_config.TextColumn("expected", help="What the ground truth asserts."),
            "content": st.column_config.NumberColumn("content", format="%.2f", help="Text found in the output (1 = yes)."),
            "structure": st.column_config.NumberColumn("structure", format="%.2f", help="Recovered as the right element, in the right place."),
            "detail": st.column_config.TextColumn("detail", help="Why it was not fully recovered.")})
        if shown.empty:
            st.success("Every expected element was recovered.")
    with tab_out:
        text = "\n\n".join(a["pages"])
        scored_pages = postprocess.apply(list(a["pages"]), list(steps)) if steps else list(a["pages"])
        versions = ["As received"] + (["After post-processing (what is scored)"] if steps else []) + ["Rendered"]
        for tab, label in zip(st.tabs(versions), versions):
            with tab:
                if label == "Rendered":
                    st.markdown(text)
                    st.caption("HTML tags are not interpreted.")
                elif label == "As received":
                    st.code(text or "(empty)", language="markdown", height=420)
                else:
                    st.code("\n\n".join(scored_pages) or "(empty)", language="markdown", height=420)
        if a.get("raw") is not None:
            with st.expander("Raw response of the service"):
                st.json(a["raw"], expanded=1)
    with tab_json:
        g1, g2 = st.columns(2)
        g1.caption("Ground truth: the elements this PDF must yield")
        g1.json([e.to_dict() for e in gt.elements], expanded=2)
        scored_pages = postprocess.apply(list(a["pages"]), list(steps)) if steps else list(a["pages"])
        g2.caption(f"Recovered from the output of {name} (converted to the same element model)")
        g2.json([e.to_dict() for e in to_elements(scored_pages)], expanded=2)


with t_detail:
    detail()

# --- About -----------------------------------------------------------------------------------------------------------

with t_about:
    st.markdown(
        "* **content**: the element's text is somewhere in the output (similarity of at least 90%).\n"
        "* **structure**: recovered as that kind of element and in the right place: a heading at the right level, a "
        "table cell at the right row and column (merged and nested cells included), a list item at the right depth, "
        "a link with its target, a footnote marked as such, a figure next to its caption, a form value next to its "
        "label, a ticked box shown as ticked.\n"
        "* Running headers and footers score structure 1 when they are absent from the output.\n"
        "* A model that returns flat text has content without structure: that gap is what this benchmark measures.\n"
        "* **numbers**: every amount, date, time and digit group of the ground truth must appear exactly as written "
        "(a wrong digit is not forgiven by the 90% text similarity). It is a content measure only.\n"
        "* **title**: the first heading of the output is the document title; 1.0 at H1, 0.5 at another level.\n"
        "* **section_path**: a paragraph, list or table sits under the right headings (the chunk's context).\n"
        "* **page**: the elements are on the right page (needs the output split by page).\n"
        "* Form fields and checkboxes are read from the text (a tick must be written ☒ ☑ ✓ or [x]). Footnotes count "
        "only when marked (`[^1]: text`). See `src/pdf_bench/score.py` and `docs/STRUCTURED.md`.")
    st.markdown("**Limits to keep in mind**")
    st.markdown(
        "* Synthetic English PDFs: they test structure, not the variety of real documents.\n"
        "* Few questions for some kinds (see **n** in *By kind*): the `paragraph` kind rests on a single PDF.\n"
        "* ★ the reference model wrote these PDFs, so its score is a ceiling, not a result.\n"
        "* The rules were refined while reading these models' outputs (and checked against an LLM judge during development).\n"
        "* Durations compare local tools with APIs; costs are list prices, not invoices.")
