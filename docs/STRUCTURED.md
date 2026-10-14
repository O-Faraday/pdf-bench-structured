# Structured-extraction benchmark: documents and ground truth

Goal: compare how well a parser recovers the **elements** of a PDF (headings, lists, tables, nested
tables, merged cells, multi-page tables, links, footnotes, figures, form fields, checkboxes, running
headers and footers, scans), whatever format its output takes.

## Generate

```bash
uv run pdf-bench synth data/structured                 # one PDF of each kind (16) + <stem>.gt.json
uv run pdf-bench synth data/structured --count 50      # 50 PDFs
uv run pdf-bench synth data/structured --count 8 --only constat,table_nested --seed 3
```

* `--count N`: number of PDFs. Kinds are taken round-robin, so `N` larger than the number of kinds
  repeats each kind with the next seed (`forms`, `forms-s01`, `forms-s02`...). With a small `N` the
  richest kinds come first (`constat`, `rapport`, `table_nested`, `table_merged`, `forms`...).
* `--seed S`: first seed (default 0). Seed 0 is the reference document. Other seeds keep the layout
  of the kind and change the content: names, amounts, dates, number of table rows, ticked boxes,
  figure values and filler text. A variant always keeps at least one ticked and one empty box, and a
  multi-page table always spans pages.
* `--only a,b`: restrict to some kinds. Kinds: `constat`, `rapport`, `table_nested`, `table_merged`,
  `forms`, `headings`, `lists`, `table_simple`, `table_borderless`, `table_multipage`, `links`,
  `footnotes`, `figures`, `columns`, `running`, `scan`.
* Every file written is checked: each text asserted by the ground truth must be readable in the PDF.

## Documents

* **One per category** (14 kinds): the ground truth lists only the element type under test; the filler
  text on the page is not asserted. `headings`, `lists`, `table_simple`, `table_borderless`,
  `table_merged`, `table_nested`, `table_multipage`, `links`, `footnotes`, `figures`, `forms`,
  `columns` (reading order), `running` (header and footer), `scan` (no text layer).
* **Composite** (2 kinds): `constat` (accident report: fields, ticked boxes, nested tables, a sketch) and
  `rapport` (headings, nested list, link, footnote, merged-header table with caption, figure, a
  table over two pages, running header and footer).

## Ground truth

`<name>.gt.json` holds `elements` in expected reading order, in a format-free element model
(`src/pdf_bench/elements.py`). Only the types listed in `tested` are asserted.

| Element | Meaning |
|---|---|
| `heading` | A title; `level` 1 is the highest. |
| `paragraph` | A block of running text. |
| `list` | A bulleted or numbered list (`ordered`); its children are `list_item`s. |
| `list_item` | One item of a list; a nested list is a child. |
| `link` | Clickable text; `href` is its target. |
| `footnote` | Text at the foot of the page, tied to a `marker` in the body. |
| `table` | A grid of `cell`s; may be nested inside a cell, may span pages. |
| `cell` | One table cell at row `r`, column `c`; `header` for a header cell; may span rows or columns. |
| `field` | A form field: `label` and `value` (a blank field has an empty value). |
| `checkbox` | A box that is ticked or not (`checked`), with its label. |
| `figure` | An image; its `caption` is a child. |
| `caption` | The legend of a figure or table. |
| `header` | Running text repeated at the top of every page. |
| `footer` | Running text repeated at the bottom of every page. |

## Honest limits

* Generated PDFs are cleaner than real ones. They are a proxy: score a few real documents annotated
  by hand before trusting a ranking.
* The scanner is simulated (skew, blur, noise, JPEG); real scans are worse.
* Form fields are plain text with a rule: a parser may legitimately output them as a table. The
  scorer must not count that as an error when `table` is not under test.
* Variants differ in content, not in layout: 100 PDFs are not 100 different page designs.
* Not covered yet: formulas, handwriting, stamps, multilingual text.
* The scorer (`score.py`) converts Markdown/HTML output to this model (`convert.py`) and compares it with the
  ground truth. Form fields, checkboxes and footnotes are read heuristically from the text (a tick must be
  written ☒ ☑ ✓ or [x]; a footnote counts when marked `[^n]: text` / `<aside data-footnote>`, or when it is its own block that starts with its number: `1 note`, `> 1 note`, `[1] note`). Native JSON outputs are reached
  through the adapters, which turn them into Markdown/HTML first.
* Only PyMuPDF4LLM and the two oracles have been run for real; the other adapters are untested against the
  actual tools and services.

Rules fixed after the first real run (2026-10-09): a paragraph counts only if a block holds the whole text (a fragment or a
short block inside it is not the paragraph; the detail says "only a fragment"); bullet or number of a list is not judged
(PDFs print numbers as text), only the items and their depth; a caption must be matched by a block as long as itself.

## Measures added after the first real run

Computed for every PDF from the existing ground truth (no new dataset needed):

| Kind | What it checks | Why |
|---|---|---|
| `numbers` | Every number of the ground truth (amounts, dates, times) appears exactly, as a multiset; a thousands group written with spaces ("12 480,50") is one value. Content only: its structure cell is blank by design. | The 90% text similarity forgives a wrong digit in a long amount ("1,234,567.89" read "1,234,767.89"). |
| `title` | The first heading of the output is the document title: 1.0 at H1, 0.5 at another level. | Docling and Databricks have a `title` type apart from section headers. |
| `section_path` | A paragraph, list or table has the right headings above it. | The context a chunk inherits; flat heading levels lose the parents. |
| `page` | The asserted headings, paragraphs, captions and footnotes are on the right page (pages read back from the PDF text layer; none for scans). | Source citation and chunking by page; fails when the output is one text. |


## Test set changes after the third run

* New document `numbers` (amounts in four styles, decimals, percentages, negatives, times, near-identical values, a phone number) so that the `numbers` measure can fail.
* Table amounts of every document now use mixed styles (`12 480,50`, `9,800.25`, `845,5`, `1 020`).
* `headings` and `rapport` assert their body paragraphs and `headings` spans two pages, so `section_path` and `page` have rows on more documents.
* Re-run `pdf-bench synth` and the models: the PDFs and ground truths changed.

## Rules fixed after reading every model's output one by one

* **Checkbox**: the state may be written after the label (`option Label: [x]`).
* **Figure**: a chart returned as data (a table or text holding its title and every label) next to the caption counts as
  recovered, with the note "chart returned as data". `[Figure: ...]`, `[Chart: ...]` markers are figures too.
* **Caption in `<div>`**: a bare `<div>text</div>` (PaddleOCR-VL) is a paragraph, no longer ignored.
* **Footnotes**: several numbered notes in one block ("1 First. 2 Second.") are read one by one.
* **Link** is *not applicable* for the models that only see page images (Claude, Gemini, GPT): the target is not in the picture.
* **Incomplete answers**: a page cut by the token limit (or refused) is an error, not a score. Run it again.
* PyMuPDF4LLM has an option `layout` (`true` default, `false` = legacy mode): legacy keeps the link targets, layout reads tables and headings better.
* LandingAI keeps footnotes, figure captions and signature text (adapter 2).
* **Docling heading levels**: Docling leaves every PDF heading at level 1 unless `HeadingHierarchyOptions(enabled=True)` is set
  (and `generate_parsed_pages=True`); levels then come from the PDF bookmarks, the numbering ("1.", "1.1", PART, ...) and the font
  style. The adapter turns it on by default (setting `heading_hierarchy`, `false` to compare). Docling answers cached before are re-run.
* **Docling tables**: settings `table_model` (`tableformer` default, `tableformer_v2`, `granite_vision`) and `cell_matching`
  (`false` lets the table model define the cells instead of snapping them to the PDF's text cells, which Docling says can break
  tables whose PDF cells span several columns). Merged cells are only kept by `format=html`. Make one named variant per
  combination (Models tab) to compare them; none was run here (the model weights are not reachable from the sandbox).

## Rules fixed after comparing them with the LLM judge

104 questions of the 1,300 asked in one run (13 models x 16 PDFs) got different scores from the rules and from the judge. Reading
each one, these were faults of the rules or the converter, now fixed (the effect on the structure score of each model is in the
commit message and the README):

* **Nested HTML tables cut in pieces** (converter): a blank line inside `<table>` ended the Markdown HTML block, so a table
  holding nested tables came out as 3 tables and 20 of 22 cells were lost. Claude's perfect nested table scored 0.09.
* **Flattened nested tables** score half for the cells they hold (the values are there, the nesting is not). Rows shifted by a
  constant (a two-level header written as one row) score half for the cells in the right columns.
* **Lists**: lines starting with a bullet glyph (`•`) are a list even when every item is its own block; Docling's flat
  `- 1 text` items are nested by their numbering restarts. Numbered lines closing a page stay footnotes.
* **Checkbox marks**: `✗ ✘ ☓` and the LaTeX `\times`, `\checkmark`, `\boxtimes` (ticked), `\square` (empty) are read.
* **Links**: a link that wraps more words than expected ("Details are in the reference document.") still carries its target.
* **Heading levels**: when every heading of a document is shifted by the same amount (Docling writes its first level as `##`) and at
  least three headings of two levels are found, the hierarchy is right and counts as such (title included).
* **Section path**: a block the tool did not mark as its kind (a list written as plain lines) still has its context.
* **Figures**: the caption may be a list item; a chart returned as data counts when three quarters of its labels and values
  sit in the short blocks (100 characters at most) around the caption.
* **Fields**: the value must come before the next field's label (Docling wrote "First name Martin Claire").
* **Paragraph content**: a paragraph cut in two blocks (two-column pages) keeps the share of it that is found.

* **Bullet glyph lines inside a paragraph block** ("Intro.\n• a\n• b") are split off as a list; a lone `•` line is a list of one item.
* **Short texts** (12 characters or fewer) must be there as whole words: "Signatures" was found in "[signature]".
* **Merged cells lost**: the right text in the right place without its span counts half (Markdown cannot express it).

Not changed, on purpose: a blank field only needs its label; a footnote written as its own block starting with its number counts;
a caption written as a heading next to the table counts. The judge is stricter on these and the rules follow the documented definition.
The judge's instructions were aligned with these definitions (prompt version 2), which also means its saved verdicts must be asked again.
