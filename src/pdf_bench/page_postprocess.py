"""Post-process page: rules applied to every model's output before scoring, with their effect shown first.

The steps live in :mod:`pdf_bench.postprocess`. They never read the ground truth. The chosen steps are saved in
the selection file and used by the Compare page and the analysis files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pdf_bench import config, postprocess, results, selection, view
from pdf_bench.elements import load_gt
from pdf_bench.score import Report  # noqa: F401  (documented type of the frames below)

config.load_env()
st.title("Post-process the outputs")
st.caption("Some structure is in the output but not written the way the scorer reads it: a table cut at a page break, "
           "a list printed as plain numbered lines, a section number that gives the heading level. These generic rules "
           "rewrite every model's output the same way; compare the scores before and after. Reads the answers already "
           "saved: nothing is run here.")

path = st.session_state.get("selection_file", str(selection.DEFAULT_PATH))
saved = selection.load(path)
chosen = st.multiselect("Steps", list(postprocess.STEPS), default=[x for x in saved.postprocess if x in postprocess.STEPS],
                        format_func=lambda k: postprocess.STEPS[k][0],
                        help="Applied in this order: merge tables, lists, heading levels.")
if st.button("Save these steps in the selection"):
    saved.postprocess = list(chosen)
    saved.save(path)
    st.success(f"Saved in {path}: the Compare page and the analysis files now score after these steps.")
elif list(chosen) != saved.postprocess:
    st.caption("Not saved yet: the Compare page still uses the saved steps.")

docs_dir = Path(saved.docs_dir)
pdfs = [p for p in sorted(docs_dir.glob("*.pdf")) if p.with_suffix(".gt.json").is_file()] if docs_dir.is_dir() else []
if saved.docs_selected:
    pdfs = [p for p in pdfs if p.name in saved.docs_selected]
answers = {}
for e in saved.entries():
    for p in pdfs:
        a = results.load_cached(e.parser, p, options=e.options or None, label=e.label) if e.parser else None
        if a and not a["error"]:
            answers[(e.label, e.parser, p)] = a
if not answers:
    st.info("No saved answer for the selection yet: run the models on the Compare page first.")
    st.stop()


def means(f: pd.DataFrame) -> float | None:
    k = f.groupby("kind")["structure"].mean().dropna()
    return round(float(k.mean()), 3) if len(k) else None


@st.cache_data(show_spinner="Scoring...")
def frame(label: str, parser: str, pdf: str, created: str, pages: tuple, steps: tuple) -> pd.DataFrame:
    return view.frame(label, parser, pdf, list(pages), steps)


def both(key):
    label, parser, pdf = key
    a = answers[key]
    return (frame(label, parser, str(pdf), a["created"], tuple(a["pages"]), ()),
            frame(label, parser, str(pdf), a["created"], tuple(a["pages"]), tuple(chosen)))


st.subheader("Effect on each model")
rows = []
for label in dict.fromkeys(k[0] for k in answers):
    pairs = [both(k) for k in answers if k[0] == label]
    raw, after = means(pd.concat([r for r, _ in pairs])), means(pd.concat([x for _, x in pairs]))
    rows.append({"model": label, "structure (raw)": raw, "structure (after)": after,
                 "gain": None if raw is None or after is None else round(after - raw, 3)})
st.dataframe(pd.DataFrame(rows).set_index("model"), width="stretch", column_config={
    c: st.column_config.NumberColumn(c, format="%.2f") for c in ("structure (raw)", "structure (after)", "gain")})

st.subheader("By kind of element")
kinds: dict[str, dict] = {}
for key in answers:
    r, x = both(key)
    for tag, f in (("raw", r), ("after", x)):
        for kind, v in f.groupby("kind")["structure"].mean().items():
            kinds.setdefault(kind, {}).setdefault((key[0], tag), []).append(v)
table = pd.DataFrame({k: {f"{m} ({t})": (None if all(pd.isna(x) for x in v) else round(float(pd.Series(v).mean()), 2))
                          for (m, t), v in d.items()} for k, d in kinds.items()}).T
table = table.loc[sorted(table.index)]
changed = table[table.apply(lambda row: any(row[c] != row[c.replace("(raw)", "(after)")] for c in table.columns if "(raw)" in c), axis=1)]
st.caption("Only the kinds a step changed (for the others, nothing differs).")
st.dataframe(changed if len(changed) else table.head(0), width="stretch")

st.subheader("One output, before and after")
c1, c2 = st.columns(2)
pick = c1.selectbox("PDF", [p.name for p in pdfs])
labels = [k[0] for k in answers if k[2].name == pick]
if not labels:
    st.info("No saved answer for this PDF.")
    st.stop()
model = c2.selectbox("Model", labels)
key = next(k for k in answers if k[0] == model and k[2].name == pick)
a = answers[key]
after = postprocess.apply(list(a["pages"]), list(chosen))
l, r = st.columns(2)
l.caption("As received")
l.code("\n\n".join(a["pages"]) or "(empty)", language="markdown", height=380)
r.caption("After the chosen steps" + ("" if chosen else " (none chosen)"))
r.code("\n\n".join(after) or "(empty)", language="markdown", height=380)
rf, xf = both(key)
delta = rf.merge(xf, on=["kind", "expected"], suffixes=(" raw", " after"))
delta = delta[delta["structure raw"].fillna(-1) != delta["structure after"].fillna(-1)]
st.markdown("**Elements whose structure score changed**")
st.dataframe(delta[["kind", "expected", "structure raw", "structure after", "detail after"]] if len(delta) else
             pd.DataFrame({"": ["no change on this PDF"]}), width="stretch", hide_index=True)
