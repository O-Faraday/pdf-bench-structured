# OpenDocRouter: models to test

[OpenDocRouter](https://www.opendocrouter.ai/) (made by LlamaIndex) is one parsing endpoint, `POST /v1/parse`, in front of
document-parsing specialists and frontier models. You pick a model id, send a PDF, and get Markdown per page. Only
successful pages are billed; new accounts get $5 of credit.

Models list read on 2026-10-09 (price version 2026-10-06). The app reads the live list, with the per-dimension
scores, from the public `GET /v1/models` (Models tab, *OpenDocRouter models*).

## Ranked by ParseBench overall

| # | Model id | ParseBench overall | $ / 1,000 pages | Kind |
|---|---|---|---|---|
| 1 | `anthropic/claude-opus-5-5` | 84.20 | 48.82 | frontier |
| 2 | `google/gemini-3-flash` | 79.70 | 19.67 | frontier |
| 3 | `openai/gpt-5.6-terra` | 75.88 | 19.89 | frontier |
| 4 | `google/gemini-3.8-flash-low` | 73.35 | 5.91 | frontier |
| 5 | `lightonai/lightonocr-3-1b` | 73.21 | 3.19 | specialist (1B) |
| 6 | `openai/gpt-6-luna` | 71.34 | 0.80 | frontier, cheap |
| 7 | `opendatalab/mineru2.5-pro` | 70.05 | 0.86 | specialist |
| 8 | `anthropic/claude-haiku-5-5` | 70.01 | 1.22 | frontier, cheap |
| 9 | `infly/infinity-parser2-flash` | 67.28 | 4.34 | specialist |
| 10 | `paddlepaddle/paddleocr-vl-1.6` | 65.50 | 2.11 | specialist |
| 11 | `rednote-hilab/dots.mocr` | 61.92 | 3.97 | specialist |
| 12 | `xingchen-agi/teleocr` | 57.25 | 2.70 | specialist |

**How much to trust this.** ParseBench is run and published by the service's own maker, not by an independent party.
The overall score mixes five dimensions (tables, charts, content faithfulness, semantic formatting, visual
grounding); the app shows each of them, and *tables*, *semantic formatting* and *visual grounding* are the closest to
what this project scores. None of them tests form fields, checkboxes, footnotes or nested/merged cells as such. Use the
ranking to choose what to test; this project's own heatmaps decide.

## Suggested first run (five models, most relevant first)

1. `anthropic/claude-opus-5-5`: the best overall, and the most expensive (about $49 per 1,000 pages).
2. `google/gemini-3-flash`: close behind on the score for 40% of the price.
3. `lightonai/lightonocr-3-1b`: the best specialist on the list; shows what a 1B model does here.
4. `opendatalab/mineru2.5-pro`: a specialist at under $1 per 1,000 pages.
5. `openai/gpt-6-luna`: the cheap frontier point (about $0.80 per 1,000 pages).

The 16 synthetic PDFs total 22 pages. At the listed averages the five models cost about $1.6 together (mostly Opus, about
$1.1), an estimate: the real charge depends on the pages. Start with `pages`-light documents if in doubt; the Compare
overview shows the actual cost per model.

## Configuration

Put the key in `.env` (never in the app or `benchmark.json`):

```
OPEN_DOC_ROUTER_API_KEY=...
```

In the app: Models tab, *OpenDocRouter models*, *List OpenDocRouter models*, sort by a dimension, tick models (or *Add
the best N*). Each becomes a **named variant** of the `opendocrouter` parser, called `odr:<model>`; confirm the
hosted warning and *Save selection*. In `benchmark.json`:

```json
{"variants": {"odr:gemini-3-flash": {"parser": "opendocrouter", "options": {"model": "google/gemini-3-flash"}}},
 "allow_hosted": true}
```

Settings of the `opendocrouter` parser:

| Setting | Default | Meaning |
|---|---|---|
| `model` | (none) | the model id, exactly as listed |
| `pages_per_request` | 50 | pages per sync request; lower it if a model's `max_sync_pages` is smaller |
| `base_url` | https://www.opendocrouter.ai | leave as is |

How the adapter calls the service: the PDF goes inline (base64) when it is under 3 MB, otherwise through the upload
endpoint; long documents are sent in page ranges; 429/503 answers are retried (`Retry-After` honoured). **A failed
page makes the whole run an error** (failed pages are not billed, and a partial answer would be scored as if the
model had missed them): run again. The Compare overview adds a **cost $** column from the `charge_usd` the service
reports for each run.

Not used: `layout` (element positions), `cache` and async mode. The documents leave your machine.

## Sources

* https://www.opendocrouter.ai/ , its documentation and `GET /v1/models` (read 2026-10-09).
