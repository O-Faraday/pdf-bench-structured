---
name: analyse-run
description: Use when the user shares or asks about the results of a benchmark run (analysis.md, analysis.json, ranking, by-kind tables) to read it, check the scores and say what is reliable.
---

# Analyse a run

1. Read the header: number of PDFs and models, missing answers, post-processing steps, evaluation fingerprint.
2. Spot incomplete models (`PDFs` below the total) and errors: a score over fewer PDFs is not comparable.
3. Set Claude apart: it generated the PDFs, so it is the reference and is never ranked.
4. Compare raw and post-processed scores: a large gain points to a "dialect" the normalisation fixes.
5. Read by kind of element, with the number of questions (`n`): conclude nothing on a kind with a small `n`.
6. Check suspicious scores: content 1.0 but structure 0.0, 0.0 on a whole kind, structure 1.0 with weak evidence
   (see the "Automatic checks" section of `analysis.md`).
7. Report: figures, what is reliable, what is not, limits (synthetic documents, small `n`, local and API durations
   not comparable, costs = public prices or billed amounts). Announce nothing that was not measured.
