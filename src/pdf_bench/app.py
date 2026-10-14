"""Streamlit app: ``uv run pdf-bench app`` (or ``uv run streamlit run src/pdf_bench/app.py``).

Two pages: *Documents and models* (create the synthetic PDFs, choose the parsers and the post-processing steps) and
*Compare* (run the chosen models, see each one against the ground truth, raw and after post-processing). Secrets stay in ``.env`` and are
never displayed or saved.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pdf_bench import config, selection
from pdf_bench.parsers import REGISTRY

config.load_env()
st.set_page_config(page_title="PDF structured extraction bench", layout="wide")
# Wide enough for the score tables, but centred with a cap: a full-width page is too wide to read or to present.
st.markdown("<style>[data-testid='stMainBlockContainer']{max-width:1180px;margin-left:auto;margin-right:auto;}</style>",
            unsafe_allow_html=True)

with st.sidebar:
    st.text_input("Selection file", str(selection.DEFAULT_PATH), key="selection_file",
                  help="Written by 'Save selection', read by the Compare page.")
    st.subheader("Credentials (.env)")
    env_file = config.load_env()
    st.caption(f".env: {env_file if env_file else 'not found (copy .env.example to .env)'}")
    for name in sorted({v for p in REGISTRY.values() for v in p.needs}):
        ok = bool(config.get(name))
        st.caption(("✅ " if ok else "⚠️ ") + name + (" set" if ok else " missing"))
    st.caption("Values are never displayed or saved.")

here = Path(__file__).parent
page = st.navigation([
    st.Page(here / "page_setup.py", title="Documents and models", icon="⚙️", default=True),
    st.Page(here / "page_compare.py", title="Compare", icon="🔎"),
])
page.run()
