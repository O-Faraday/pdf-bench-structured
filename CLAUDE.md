# pdf-bench

## Goal
Benchmark of **structured** PDF extraction: synthetic PDFs with an element-level ground truth (headings, lists, tables,
links, footnotes, figures, form fields, checkboxes, running headers/footers, scans) measure which parser keeps the
structure that chunking, embeddings and retrieval need. A Streamlit app compares the models.

## Language
Answer in French. Code, comments, commit messages and the repository's documentation are in English.

## Stack
Python 3.12, `uv` (`pyproject.toml`, `uv.lock`), Streamlit, pandas, pymupdf, pytest.
Parsers: pymupdf4llm, Docling, OpenDataLoader (local); Claude, Gemini, GPT, Mistral OCR, Databricks
`ai_parse_document`, LandingAI ADE, LlamaParse, Unstructured, OpenDocRouter (hosted).

## Structure
| Path | Role |
|---|---|
| `src/pdf_bench/synth.py`, `elements.py` | PDF generation and element model (ground truth `*.gt.json`) |
| `src/pdf_bench/convert.py`, `locate.py` | a parser's Markdown/HTML output -> elements; pages read back from the PDF |
| `src/pdf_bench/score.py` | deterministic scoring: one row per element, `content` and `structure` |
| `src/pdf_bench/postprocess.py` | normalisation steps shared by every model (they never read the ground truth) |
| `src/pdf_bench/view.py`, `analysis.py`, `recommend.py` | scores as the app shows them, analysis files, recommendation |
| `src/pdf_bench/results.py`, `selection.py`, `pricing.py` | answer cache, `benchmark.json` selection, costs |
| `src/pdf_bench/judge.py`, `questions.py`, `rerun.py` | LLM judge (command-line calibration tool), questions, re-evaluation without parsing |
| `src/pdf_bench/parsers/` | one module per parser; `base.py` = registry |
| `src/pdf_bench/app.py`, `page_setup.py`, `page_compare.py` | Streamlit app: choose documents/models/steps, compare |
| `docs/STRUCTURED.md` | documents, ground truth, scoring rules |
| `tests/` | pytest |

## Commands
- Install: `uv sync`
- Tests: `uv run pytest -q`
- App: `uv run pdf-bench app`
- Test documents: `uv run pdf-bench synth data/structured`
- Score the saved answers again (no parsing): `uv run pdf-bench evaluate` (alias `analysis`)
- LLM judge, calibration: `uv run pdf-bench evaluate --judge` (costs tokens)
- Add OpenDocRouter models: `uv run pdf-bench add-odr <model-id>`
- Parsers available here: `uv run pdf-bench parsers`

## Conventions
- Read a library's current documentation before writing code that calls it.
- A new parser option goes through `config.py` and gets an entry in the README (`.env.example` if it is a secret).
- A behaviour change gets a test in `tests/`. Run `uv run pytest -q` before concluding.
- A scoring-rule change is documented in `docs/STRUCTURED.md`.
- The answer cache depends on the PDF, the parser, its version and its options: do not bypass `results.py`.
- Models and their scores stay in `results/` (not committed); never edit them by hand.

## Never
- No API key or token on disk, in a commit, a log or a selection file: `.env` only.
- The steps of `postprocess.py` never read the ground truth.
- No score gain announced without measuring it (run the evaluation again, compare before and after).
- Claude generated the PDFs: it is a reference, never ranked or recommended (`recommend.REFERENCE_MODELS`).
- Do not send PDFs to a third-party service without the user's confirmation (`hosted` parsers).

## Method: analysing a run
See the `analyse-run` skill (`.claude/skills/analyse-run/SKILL.md`).
