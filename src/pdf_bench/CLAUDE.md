# src/pdf_bench

- Scoring (`score.py`) is deterministic and is tuned against the ground truth, never against a model's name.
- `view.frame` is the only place that turns a parser's pages into scored rows: the app, the CLI and the analyses
  must go through it.
- After changing the scorer, the converter or the post-processing, evaluate again with `rerun.reevaluate()` (the
  "Rerun evaluation" button of the app) or `uv run pdf-bench evaluate`, without parsing again.
- A parser declares what it cannot express in `cannot` ("not applicable") instead of getting a 0.
