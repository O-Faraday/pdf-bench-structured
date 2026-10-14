"""Setup page: create the synthetic documents, then choose the parsers to compare.

The choice is saved to ``benchmark.json`` and read by the comparison page. Secrets stay in ``.env`` and
are never shown or saved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pymupdf
import streamlit as st

from pdf_bench import config, postprocess, selection
from pdf_bench.elements import load_gt
from pdf_bench.parsers import (
    CHECKED,
    FAMILIES,
    REGISTRY,
    databricks_util,
    opendocrouter_util,
)
from pdf_bench.synth import KINDS, check_gt, generate_structured, plan

config.load_env()
st.title("Documents and models")

SELECTION_FILE = st.session_state.get("selection_file", str(selection.DEFAULT_PATH))
saved = selection.load(SELECTION_FILE)

tab_docs, tab_models = st.tabs(["1 · Documents", "2 · Models"])

# --- documents -----------------------------------------------------------------------------------------

with tab_docs:
    c1, c2, c3 = st.columns([2, 1, 1])
    docs_dir = c1.text_input("Output folder", saved.docs_dir)
    one_each = c2.checkbox("One of each kind", value=saved.docs_count is None,
                           help=f"{len(KINDS)} kinds. Untick to choose how many PDFs to create.")
    count = None if one_each else int(c2.number_input("Number of PDFs", 1, 1000, saved.docs_count or 50))
    seed = int(c3.number_input("First seed", 0, 10_000, saved.docs_seed,
                               help="Seed 0 is the reference document; other seeds change the content, not the layout."))
    kinds = st.multiselect("Kinds", list(KINDS), default=saved.docs_only or list(KINDS),
                           help="Round-robin over these kinds. The first kinds are the richest.")
    todo = plan(count, seed, kinds) if kinds else []
    if todo:
        per_kind = pd.Series([k for k, _ in todo]).value_counts().rename("PDFs")
        st.caption(f"{len(todo)} PDF(s) will be written to `{docs_dir}`: " +
                   ", ".join(f"{k} ×{n}" for k, n in per_kind.items()))
    else:
        st.warning("Select at least one kind.")
    if st.button("Generate PDFs", type="primary", disabled=not todo):
        with st.spinner(f"Writing {len(todo)} PDF(s)..."):
            paths = generate_structured(docs_dir, count=count, seed=seed, only=kinds)
            problems = []
            for pdf in paths:
                problems += check_gt(pdf, load_gt(pdf.with_suffix(".gt.json")))
        st.session_state["generated"] = [str(p) for p in paths]
        (st.error if problems else st.success)(
            f"{len(paths)} PDF(s) written; ground truth checked against the PDFs: "
            + (f"{len(problems)} problem(s), first: {problems[0]}" if problems else "consistent."))

    existing = sorted(Path(docs_dir).glob("*.pdf")) if Path(docs_dir).is_dir() else []
    selected_docs: list[str] = []
    if existing:
        st.markdown("**PDFs to run the models on**")
        chosen_before = set(saved.docs_selected) or {p.name for p in existing}
        rows = []
        for p in existing:
            gp = p.with_suffix(".gt.json")
            g = load_gt(gp) if gp.is_file() else None
            with pymupdf.open(p) as d:
                n_pages = len(d)
            rows.append({"run": p.name in chosen_before, "pdf": p.name, "category": g.category if g else "no ground truth",
                         "asserts": ", ".join(g.tested) if g else "", "pages": n_pages})
        table = pd.DataFrame(rows)
        edited_docs = st.data_editor(
            table, hide_index=True, width="stretch", key=f"doc_table_{docs_dir}_{len(existing)}",
            disabled=[c for c in table.columns if c != "run"],
            column_config={"run": st.column_config.CheckboxColumn("run", help="Tick to run the selected models on this PDF."),
                           "category": st.column_config.TextColumn("category", help="Kind of document in the ground truth.")})
        selected_docs = edited_docs.loc[edited_docs["run"], "pdf"].tolist()
        st.caption(f"{len(selected_docs)} of {len(existing)} PDF(s) selected. Press *Save selection* on the Models tab.")
        pick = st.selectbox("Preview", [p.name for p in existing])
        pdf = Path(docs_dir) / pick
        gt_path = pdf.with_suffix(".gt.json")
        left, right = st.columns([1, 1])
        with pymupdf.open(pdf) as doc:
            left.image(doc[0].get_pixmap(dpi=70).tobytes("png"), caption=f"{pick} (page 1 of {len(doc)})")
        if gt_path.is_file():
            gt = load_gt(gt_path)
            right.markdown(f"**Ground truth**: category `{gt.category}`, asserts " + ", ".join(f"`{t}`" for t in gt.tested))
            right.json(gt.counts(), expanded=True)
            rows = [{"type": e.type, "text": e.text[:70], **{k: v for k, v in e.attrs.items() if k != "image_text"}}
                    for e in gt.walk() if e.type not in ("cell",)]
            right.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=300)
    else:
        st.info("No PDF in this folder yet.")

# --- models -------------------------------------------------------------------------------------------

with tab_models:
    st.caption(f"Catalogue checked on {CHECKED}. Only parsers that recover structure are listed.")
    f1, f2 = st.columns([3, 1])
    families = f1.multiselect("Families", list(FAMILIES), default=list(FAMILIES), format_func=FAMILIES.get)
    only_ready = f2.toggle("Only available here", value=True,
                           help="Hide parsers that cannot run now (not installed, no credentials, no server).")

    rows = []
    for p in REGISTRY.values():
        status = p.status()
        if p.family not in families or (only_ready and status):
            continue
        rows.append({"run": p.name in saved.parsers, "parser": p.name, "label": p.label or p.name,
                     "family": p.family, "status": status or "ready", "hosted": p.hosted, "gpu": p.gpu,
                     "license": p.license, "version": p.version, "reference": p.reference, "tested": p.tested})
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No parser to show: untick 'Only available here' to see why the others cannot run.")
        chosen: list[str] = []
    else:
        edited = st.data_editor(
            df, hide_index=True, width="stretch", key="parser_table",
            disabled=[c for c in df.columns if c != "run"],
            column_config={
                "run": st.column_config.CheckboxColumn("run", help="Tick to include in the benchmark."),
                "hosted": st.column_config.CheckboxColumn("hosted", help="The PDFs leave your machine and usage is billed."),
                "gpu": st.column_config.CheckboxColumn("gpu", help="A GPU is needed or strongly recommended."),
                "tested": st.column_config.CheckboxColumn(
                    "tested", help="Run for real by this project. Untested adapters follow the tool's documentation."),
                "reference": st.column_config.TextColumn(
                    "reference", help="Score published elsewhere, on another benchmark: orientation only.")})
        chosen = edited.loc[edited["run"], "parser"].tolist()
    untested = [n for n in chosen if not REGISTRY[n].tested]
    if untested:
        st.warning("Never run against the real tool or service in this project: " + ", ".join(untested)
                   + ". Expect to fix an adapter on the first run.")

    # Databricks: discover warehouses and volumes
    if "databricks_ai_parse" in REGISTRY:
        creds = databricks_util.credentials_status()
        with st.expander("Databricks workspace", expanded="databricks_ai_parse" in chosen):
            if not all(creds.values()):
                st.info("Set " + " and ".join(k for k, v in creds.items() if not v)
                        + " in .env, then restart the app.")
            else:
                st.caption("DATABRICKS_HOST and DATABRICKS_TOKEN are set.")
                if st.button("Connect and list warehouses"):
                    try:
                        st.session_state["dbx_client"] = databricks_util.client()
                        st.session_state["dbx"] = databricks_util.connect(st.session_state["dbx_client"])
                    except RuntimeError as exc:
                        st.session_state.pop("dbx", None)
                        st.error(str(exc))
                ws = st.session_state.get("dbx")
                w = st.session_state.get("dbx_client")
                if ws:
                    st.success(f"Connected to {ws.host} as {ws.user}.")
                    if ws.warehouses:
                        labels = {x["id"]: f"{x['name']} ({x['state'].lower()}, {x['type'].lower()} {x['size']})".strip()
                                  for x in ws.warehouses}
                        wid = st.selectbox("SQL warehouse", list(labels), format_func=labels.get,
                                           help="ai_parse_document runs on this warehouse.")
                        st.session_state["opt_databricks_ai_parse_warehouse_id"] = wid
                    else:
                        st.warning("No SQL warehouse visible to this token.")
                    try:
                        cats = databricks_util.catalogs(w)
                        cat = st.selectbox("Catalog", cats) if cats else None
                        sch = st.selectbox("Schema", databricks_util.schemas(w, cat)) if cat else None
                        vols = databricks_util.volumes(w, cat, sch) if sch else []
                        vol = st.selectbox("Volume (the PDFs are uploaded here, then deleted)", vols) if vols else None
                        if vol:
                            st.session_state["opt_databricks_ai_parse_volume"] = vol
                        elif sch:
                            st.warning("No volume in this schema: create one, or pick another schema.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not list Unity Catalog objects: {type(exc).__name__}: {exc}")
                st.caption("The service needs Databricks Runtime 17.3+ or serverless SQL, in a supported region.")

    # per-parser settings
    options: dict[str, str] = {}
    for name in chosen:
        p = REGISTRY[name]
        with st.expander(f"Settings · {p.label or name}", expanded=bool(p.status())):
            if p.status():
                st.warning(p.status())
            if p.install:
                st.code(p.install, language="bash")
            if p.note:
                st.caption(p.note)
            if p.source:
                st.caption(f"Source: {p.source}")
            keys = dict(p.options)
            for key, default in keys.items():
                state_key = f"opt_{name}_{key}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = saved.options.get(f"{name}.{key}") or p.option(key, str(default))
                value = st.text_input(key, key=state_key)
                if value != str(default):
                    options[f"{name}.{key}"] = value

    # OpenDocRouter: pick models from its public list; each becomes a named variant of the opendocrouter parser
    odr_variants: dict[str, dict] = {}
    if "opendocrouter" in REGISTRY:
        kept = {v["options"].get("model", ""): k for k, v in saved.variants.items() if v.get("parser") == "opendocrouter"}
        with st.expander("OpenDocRouter models", expanded=bool(kept)):
            st.caption("Public list (no key needed) with the ParseBench scores the service publishes for each model. "
                       "Each ticked model becomes a named variant of `opendocrouter`, called `odr:<model>`. The scores "
                       "come from the service itself: use them to choose what to test, not to rank.")
            rank_by = st.selectbox("Sort by", list(opendocrouter_util.SCORE_LABELS.values()), index=0,
                                   help="ParseBench dimension. 'tables', 'formatting' and 'grounding' are the closest to this benchmark.")
            if st.button("List OpenDocRouter models"):
                try:
                    st.session_state["odr_payload"] = opendocrouter_util.fetch()
                except Exception as exc:  # noqa: BLE001
                    st.session_state.pop("odr_payload", None)
                    st.error(f"Could not read the model list: {type(exc).__name__}: {exc}")
            payload = st.session_state.get("odr_payload")
            if payload:
                rows = opendocrouter_util.models(payload, rank_by=rank_by)
                by_id = {r["id"]: r for r in rows}
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                    "usd_per_1k_pages": st.column_config.NumberColumn("$ / 1k pages", help="Average charge per 1,000 pages on the benchmark."),
                    **{c: st.column_config.NumberColumn(c, format="%.1f") for c in list(opendocrouter_util.SCORE_LABELS.values())}})
                st.session_state.setdefault("odr_ids", [i for i in kept if i in by_id])
                top = st.number_input("Add the best N by the chosen score", 1, max(1, len(rows)), min(5, len(rows)))
                if st.button("Add them"):
                    scored = [r["id"] for r in rows if r.get(rank_by) is not None][: int(top)]
                    st.session_state["odr_ids"] = list(dict.fromkeys(st.session_state["odr_ids"] + scored))
                ids = st.multiselect(f"Models ({len(rows)} listed)", list(by_id), key="odr_ids")
            else:
                ids = list(kept)  # list not loaded: keep what was saved
                if kept:
                    st.caption("Kept from the saved selection: " + ", ".join(kept.values()))
            for i in ids:
                odr_variants[opendocrouter_util.label(i)] = {"parser": "opendocrouter", "options": {"model": i}}

    # named variants: the same parser under other settings, each with a name to refer to in the tables
    st.markdown("**Named variants**")
    st.caption("Run one parser under several settings (e.g. two model ids) and refer to each by its name in the "
               "tables. Settings are written `key=value; key=value` (keys are the ones listed in the parser's "
               "settings above; secrets stay in `.env`).")
    base = pd.DataFrame([{"name": k, "parser": v.get("parser", ""),
                          "settings": "; ".join(f"{a}={b}" for a, b in v.get("options", {}).items())}
                         for k, v in saved.variants.items() if v.get("parser") != "opendocrouter" or "opendocrouter" not in REGISTRY],
                        columns=["name", "parser", "settings"])
    edited_v = st.data_editor(base, num_rows="dynamic", hide_index=True, width="stretch", key="variant_table",
                              column_config={"name": st.column_config.TextColumn("name", help="How this variant is called everywhere."),
                                             "parser": st.column_config.SelectboxColumn("parser", options=list(REGISTRY)),
                                             "settings": st.column_config.TextColumn("settings", help="key=value; key=value")})
    variants: dict[str, dict] = {}
    for _, r in edited_v.iterrows():
        label = str(r["name"] or "").strip()
        if not label:
            continue
        opts = {}
        for part in str(r["settings"] or "").split(";"):
            if "=" in part:
                k, _, v = part.partition("=")
                opts[k.strip()] = v.strip()
        variants[label] = {"parser": r["parser"], "options": opts}
    variants.update(odr_variants)
    hosted = [n for n in chosen + [v["parser"] for v in variants.values()] if n in REGISTRY and REGISTRY[n].hosted]
    hosted = list(dict.fromkeys(hosted))
    allow = False
    if hosted:
        allow = st.checkbox(f"I understand that {', '.join(hosted)} send the benchmark PDFs to a third party and "
                            "cost money (one request per page or document).", value=saved.allow_hosted)

    steps = st.multiselect("Post-processing steps", list(postprocess.STEPS),
                           default=[x for x in saved.postprocess if x in postprocess.STEPS],
                           format_func=lambda k: postprocess.STEPS[k][0],
                           help="Generic rules applied to every model's output before scoring (they never read the ground truth). "
                                "Applied in this order: merge tables, lists, heading levels. The Compare page shows the raw score "
                                "next to the processed one, and what each step adds.")
    sel = selection.Selection(parsers=chosen, options=options, allow_hosted=allow, docs_dir=docs_dir,
                              docs_count=count, docs_seed=seed, docs_only=[] if kinds == list(KINDS) else kinds,
                              docs_selected=[] if len(selected_docs) == len(existing) else selected_docs, variants=variants,
                              postprocess=list(steps))
    problems = sel.problems()
    if existing and not selected_docs:
        problems.append("select at least one PDF (Documents tab)")
    st.divider()
    st.markdown(f"**{len(sel.entries())} model(s) selected**: " + (", ".join(e.label for e in sel.entries()) or "none"))
    for msg in problems:
        st.warning(msg)
    if st.button("Save selection", type="primary", disabled=bool(problems)):
        path = sel.save(SELECTION_FILE)
        st.success(f"Saved to {path}. The scoring stage reads this file.")
        st.json({"parsers": sel.parsers, "options": sel.options, "variants": sel.variants, "documents": sel.docs_dir,
                 "pdfs": sel.docs_selected or "all"})
