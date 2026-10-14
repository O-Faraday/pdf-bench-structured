# pdf-bench: structured PDF extraction benchmark

Compare how well parsers recover the **elements** of a PDF (headings, lists, tables, nested tables,
merged cells, multi-page tables, links, footnotes, figures, form fields, checkboxes, running headers
and footers, scans), whatever format their output takes.

This repository currently holds the first stage: **synthetic documents with a ground truth**, and the
catalogue of the structured parsers to compare. The scorer (parser output to element model, then
content and structure scores per element) is the next stage.

```bash
uv sync
uv run pdf-bench synth data/structured               # one PDF of each of the 16 kinds
uv run pdf-bench synth data/structured --count 50    # 50 PDFs: kinds cycle, content changes with the seed
uv run pdf-bench parsers                             # which structured parsers can run here, and why not
uv run pdf-bench app                                 # the app: create documents, choose the models
uv run pytest
```

## The app

`uv run pdf-bench app` opens a multipage app (the sidebar holds the selection file and the credentials status).
Page *Documents and models* has two tabs.

1. **Documents**: output folder, number of PDFs (or one of each kind), first seed, kinds; *Generate PDFs*
   writes them, checks every ground truth against its PDF, and previews any PDF next to its ground truth.
2. **Models**: the catalogue with a status per parser (ready, or why not). *Only available here* hides what
   cannot run now. Tick the parsers, edit their settings (model id, base URL...), confirm hosted ones, and
   *Save selection*: it writes `benchmark.json` (names and non-secret options only), read by the scoring stage.

Databricks: put `DATABRICKS_HOST` and `DATABRICKS_TOKEN` in `.env`, restart the app, open *Databricks
workspace* and press *Connect and list warehouses*: pick a SQL warehouse, then a catalog, schema and volume
(the PDFs are uploaded there for `ai_parse_document`, then deleted). Without the app, set
`PDF_BENCH_DATABRICKS_AI_PARSE_VOLUME` and `DATABRICKS_WAREHOUSE_ID`.
The connection and listing code was tested against a fake workspace only, not a real one.

On the *Documents* tab, tick the PDFs to use (table with a `run` box per PDF); on the *Models* tab, tick the models and
*Save selection* (both go to `benchmark.json`).

**OpenDocRouter**: see `docs/OPENDOCROUTER.md` for the ranked models and the settings. Put
`OPEN_DOC_ROUTER_API_KEY` in `.env`; on the Models tab, *OpenDocRouter models* lists the models with their ParseBench scores
(public list), and each ticked model becomes a named variant `odr:<model>`. Hosted: the PDFs leave your machine, billed per
page; the Compare overview shows the cost.

**Named variants** (Models tab): the same parser under other settings, e.g. two model ids, each with a name (`claude-big`
= `claude`, `model=...`). The name is how the model is called in the tables, heatmaps and `results/` answers.

Page **Compare**: it shows that saved selection, then *Run the selected models* runs every selected model on every
selected PDF. You first get the **score table per model** (content and structure over every PDF, then
seaborn heatmaps by kind of element and by PDF). Then pick any **PDF and model** for the detail: scores by kind, the verdict on every
expected element, the model's output (text and rendered), and the ground-truth JSON next to the JSON of the elements
recovered from that output. Answers are cached in `results/<dataset path>/<model>/` (for example `results/data/structured_whole_model/claude/`; key: PDF bytes, parser, version, options); failures
are not cached; *Ignore saved answers* forces a new run.

Scores (`src/pdf_bench/score.py`): **content** = the text is found anywhere; **structure** = recovered as the right
kind of element in the right place. Any output (Markdown, HTML) is first converted to the element model
(`convert.py`). Two references (not parsers; they are not shown on the Compare page) run without any model: `oracle_html` (1.0 by construction) and `oracle_markdown`
(the ceiling of plain Markdown: nested and merged cells are lost).

See `docs/STRUCTURED.md` for the documents, the ground-truth format and the limits.

## Parsers (`uv run pdf-bench parsers`)

Only parsers that recover structure are listed (plain text extractors are out of scope), declared in
`src/pdf_bench/parsers/<parser>.py` (one module per parser) with version, licence, source and a published reference score.

| Family | Parsers |
|---|---|
| Local libraries (CPU) | PyMuPDF4LLM (defaults and max settings), Docling, Unstructured, **OpenDataLoader PDF** (needs Java 11+) |
| Hosted models | Claude, Gemini, GPT, Mistral OCR, **OpenDocRouter** (many parsing models behind one API) |
| Cloud document services | **Databricks `ai_parse_document`**, **LandingAI ADE**, **LlamaParse** |

* Credentials and options come from `.env` (see `.env.example`) or `PDF_BENCH_<PARSER>_<OPTION>`.
* Hosted models and cloud services send the PDFs to a third party and cost money. They are never in
  the default set and must be requested by name.
* **Only PyMuPDF4LLM has been run for real here.** Every other adapter is written from the tool's
  documentation and tested with fake clients; none was run against the real tool or service (no GPU,
  network access or credentials where it was written). Hosted model ids and Hugging Face repository
  ids are editable defaults to verify before a costly run.
* LlamaParse and Unstructured: Unstructured elements are converted to Markdown/HTML (titles by `category_depth`, tables as
  HTML). The LlamaParse SDK signatures were checked against the installed package, not against the service.
* Azure Document Intelligence and GROBID were written, then set aside; they are in the git history (commit cce5bde, "Adapters:
  Azure Document Intelligence, LlamaParse, Unstructured, GROBID").
* Databricks and LandingAI adapters write the services' typed elements as Markdown with HTML tables;
  running headers/footers are dropped and figures written as `[image]`. Databricks gives no heading
  level (titles become `#`, section headers `##`).

## OpenDataLoader PDF

Run for real on the 16 synthetic PDFs (local mode, version 2.5.12, Java 21). Its HTML output is used by default
because it keeps `rowspan`/`colspan`; Markdown (`format` setting) loses merged and nested cells.

| Output | merged table | nested table | lists | links | footnotes | figures | borderless table |
|---|---|---|---|---|---|---|---|
| HTML (default) | 1.00 | 1.00 | 0.43 | 0 | 0 | 0 | 0 |
| Markdown | 0.89 | 0.24 | 0.29 | 0 | 0 | 0 | 0 |

(structure score per PDF). Hybrid mode: make a named variant with the setting `hybrid` = `docling-fast` (Models tab). The app starts the local
hybrid server (`opendataloader-pdf-hybrid`, port 5002) by itself, or uses the one at `hybrid_url`; the first start loads
the Docling models (minutes). A failed page is an error, never a silent fallback to local mode. Not run to the end here:
the sandbox cannot download the Docling models. The output is a single text for the
whole document, not split by page.

## Analysis files (to critique the scores)

After a run, `uv run pdf-bench analysis` (or the *Write the analysis files* button on the Compare page) reads the saved
answers of the current selection and writes `analysis/<dataset path>/analysis.md` and `analysis.json`. The report has
the setup, the scores, automatic doubts about the scores themselves (text found everywhere but structure never
recovered, structure 1.0 with a weak match, contradictions, empty outputs, identical outputs), then for each PDF and model
the elements not fully recovered with the evidence found in the output, and the output itself. The JSON adds the ground
truth, the full outputs and the recovered elements. Give both files to Claude to get a critique of the scores and of the
extractions. Nothing is run by this command.

## Changing the evaluation without parsing again, and the LLM judge

Parsing is the slow, billed part; evaluating is not. Answers are saved in `results/`, so you can change how they are
judged and score the same answers again:

* **Compare page, *Rerun evaluation***: reloads the scorer, converter, post-processing and question modules
  from disk and clears the score caches. Edit `score.py` (or `convert.py`, `postprocess.py`...), press the button, no restart,
  no parser called, no service billed.
* **Command line**: `uv run pdf-bench evaluate` (same as `analysis`) re-scores the saved answers and rewrites the analysis files.

**LLM judge (`judge.py`).** Every row of a score is a *question* about the output ("is this H2 there? are these 26 cells in
the right row and column? is the box ticked?"; `questions.py` writes it from the ground truth). The rules answer it
deterministically; the judge answers the same question in words, from the same text (after post-processing), and gives
`judge_content`, `judge_structure` and a one-line `judge_reason`. Then, for every question where the two differ by 0.25 or more (structure or content), a second pass shows the judge both scores and both explanations and it rules: `rules` right, `judge` right or `neither`, with the score it holds correct and, if a rule is at fault, a one-line fix (`arb_*` columns; untick *The judge rules on its disagreements* to skip it). Changing the judge's instructions (`PROMPT_VERSION`) asks every question again. The first pass stays blind to the rules, so the ruling cannot bias it. The *Rules against
the LLM judge* part of the analysis file shows, per model and per kind, the mean by rules and by judge, the share of
questions where they agree (gap below 0.25), Cohen's kappa, and every disagreement with both explanations: each one is
either a rule to fix or a judge error. The judge never sees the rules or the PDF, only the output and the expected element.

* `uv run pdf-bench evaluate --judge` writes `analysis/<dataset path>/judge_analysis.md` and `.json` from the saved verdicts (nothing is sent): coverage, agreement per model, kind and PDF, and every disagreement with the rules' and the judge's reasons.  Give it to Claude to decide, case by case, who is right.
* The judge is no longer in the app: it is a command-line calibration tool (`uv run pdf-bench evaluate --judge`; `--judge-cached` only adds saved verdicts). It served to find and fix scorer faults (rules and judge now agree on 95-100% of questions per model); the app shows the deterministic scores only.
  Questions are sent 20 per call; tokens are reported. Nothing is sent until you ask.
* Verdicts are saved by content in `results/judge/`: the same output, questions, judge model and prompt cost nothing the second
  time. Change `PROMPT_VERSION` in `judge.py` when you change the instructions.
* Configure it in `.env`: `PDF_BENCH_JUDGE_PROVIDER` (`anthropic` default, or `openai`, also for any OpenAI-compatible
  server with `PDF_BENCH_JUDGE_BASE_URL`), `PDF_BENCH_JUDGE_MODEL`. Use a model that is not under test, or say so.
* Limits: text only (it can be fooled by output that looks right), `page` rows are not judged, kinds "not applicable" are
  not judged on structure, and a judge is an opinion: it measures how much the rules can be trusted, it does not replace the ground truth.

## Post-processing, "not applicable" and tables over several pages

* **Post-processing** (chosen on the *Documents and models* page): generic rules rewrite every model's output before scoring (`src/pdf_bench/postprocess.py`):
  drop running headers/footers repeated on several pages, merge a table cut at a page break, turn plain numbered or
  bulleted lines into lists, set the heading level from the section number. The steps never read the ground truth.
  Tick them with the selection and save it: Compare and the analysis files then score after them and show the raw
  score next to it, the gain, and (optional table) what each step adds per model.
* **Not applicable**: a tool whose documented output has no such element (Databricks: no list or link element;
  OpenDataLoader: no footnote or link element) gets *not applicable* for that kind, left out of its mean and grey in the
  heatmaps, instead of 0. It comes from the tool's documentation (`cannot` in the parser's module under `src/pdf_bench/parsers/`), never from observed
  results. A step that builds the element from text (lists) lifts the mark.
* **Tables over several pages**: the cells are scored over all the pieces (a repeated header is ignored); whether the
  tool wrote ONE table is a separate kind, `table_continuity`, which the *merge tables* step can repair.
* **Docling**: setting `format` = `html` keeps merged cells (rowspan/colspan); the default Markdown export cannot express them.
  Make a named variant to compare both.


## Cost, duration and recommendation (Compare page)

* Table *2 · Scores per model* gives, per model, the **total duration**, the **estimated cost** and how it was made
  (`cost basis`). Local tools are free; Mistral OCR, Databricks `ai_parse_document`, LlamaParse and LandingAI use the
  public per-page rates in `src/pdf_bench/pricing.py` (read on 2026-10-09, with their sources: they change). Claude, Gemini
  and GPT are billed per token: set `PDF_BENCH_<PARSER>_PRICE_IN_PER_M` and `PDF_BENCH_<PARSER>_PRICE_OUT_PER_M` in `.env`
  (USD per million tokens); the token counts are recorded with each answer, so runs cached before this version must be
  run again. A model with no known rate shows an empty cost, never a guess. Databricks is priced at the upper bound of its published range.
* Section *4 · Recommendation* lists the top 3 **free** and top 3 **paid** models on the structure of tables, figures and
  captions, form fields and checkboxes plus the text read from scans (`src/pdf_bench/recommend.py`). Only models that ran on
  every selected PDF without error are ranked; close scores are flagged as not distinguishable.


## Layout of the parsers

Each parser is one module in `src/pdf_bench/parsers/`: its adapter (PDF in, one Markdown or HTML string per page out) and its
`entry(...)` declaration (label, family, version, licence, source, settings, what it cannot express, notes).

| module | what it holds |
|---|---|
| `base.py` | the `Parser` class, `REGISTRY`, `entry()`, readiness checks |
| `common.py` | shared by the page-image models and the service writers (`Pages`, `render_pages`, `strip_fence`...) |
| `pymupdf4llm.py`, `docling.py`, `unstructured.py`, `opendataloader.py` | local libraries |
| `claude.py`, `gemini.py`, `gpt.py`, `mistral_ocr.py`, `opendocrouter.py` | hosted models |
| `databricks.py`, `landingai.py`, `llamaparse.py` | cloud document services |
| `oracles.py` | the references (ground truth as HTML or Markdown), not parsers |
| `databricks_util.py`, `opendocrouter_util.py`, `html_util.py` | helpers of the app and of one adapter |

To add a parser: copy a module of the same family, change the adapter and the `entry(...)`, and add the import to
`parsers/__init__.py` (the order of the imports is the order of the app).


## The reference model, and adding OpenDocRouter models

`claude` wrote the test PDFs, so for this dataset it is a reference point (close to a training set), not a
contender: it is run and shown, flagged in the analysis, and never ranked or advised. A model that did not
write the PDFs would be ranked normally; to test Claude fairly, use PDFs it did not create.

Add models served by OpenDocRouter without opening the app's picker (needs `OPEN_DOC_ROUTER_API_KEY` in `.env`):

    uv run pdf-bench add-odr mistral/mistral-ocr-4-1-annotations openai/gpt-5.6-terra
    uv run pdf-bench app      # Run: only the missing answers are parsed, the others come from the cache


## The Compare page

A header gives the fingerprint of the evaluation code (`Evaluation 1a2b3c4d`), the post-processing steps and the
dates of the answers: same fingerprint, same rules. *Selection and run* is a collapsible panel. Tabs:

* **Ranking**: structure and content per model, with **raw structure** and the **post-processing gain** side by side, the
  reference model starred and listed first, an optional table of what each post-processing step adds per model,
  the recommendation, and CSV / JSON export.
* **By kind**: heatmaps by kind of element (with **n**, the number of questions behind each row) and by PDF, for
  structure or content, raw or after post-processing; PNG and CSV download.
* **Detail**: one PDF against one model: elements not recovered, output as received / after post-processing / rendered, JSON.
* **About the scores**: what the numbers mean and the limits to state when publishing them.
