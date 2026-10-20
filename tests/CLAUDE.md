# Tests

- `uv run pytest -q`; an app test uses `streamlit.testing.v1.AppTest` on a page of `src/pdf_bench/`.
- No network call and no API key: hosted parsers are faked.
- Test PDFs are generated with `generate_structured` in a temporary folder (`tmp_path`).
- A scoring fix has its test in `test_scorer_fixes.py`.
