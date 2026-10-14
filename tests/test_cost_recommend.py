import pandas as pd

from pdf_bench import pricing, recommend


def opt(**kw):
    return lambda k, d="": kw.get(k, d)


def test_local_tools_are_free():
    c = pricing.estimate("docling", opt(), [], 10)
    assert c.usd == 0.0 and c.free


def test_per_page_rates():
    assert pricing.estimate("mistral_ocr", opt(), [], 100).usd == 0.4          # $4 per 1,000 pages
    assert pricing.estimate("landingai_ade", opt(), [], 10).usd == 0.3          # 3 credits x $0.01 per page
    assert pricing.estimate("llamaparse", opt(tier="agentic"), [], 100).usd == 1.25   # 10 credits x $1.25/1000
    assert pricing.estimate("llamaparse", opt(tier="agentic_plus"), [], 100).usd == 5.625
    assert "ai_parse_document" in pricing.estimate("databricks_ai_parse", opt(), [], 100).basis


def test_token_models_need_a_price_and_usage():
    ans = [{"raw": {"usage": {"input_tokens": 2_000_000, "output_tokens": 500_000}}}]
    assert pricing.estimate("claude", opt(), ans, 5).usd is None
    c = pricing.estimate("claude", opt(price_in_per_m="3", price_out_per_m="15"), ans, 5)
    assert c.usd == 13.5
    assert pricing.estimate("claude", opt(price_in_per_m="3", price_out_per_m="15"), [{"raw": None}], 5).usd is None


def test_billed_amount_wins_and_unknown_has_no_estimate():
    assert pricing.estimate("opendocrouter", opt(), [{"raw": {"charge_usd": 0.02}}, {"raw": {"charge_usd": 0.01}}], 3).usd == 0.03
    assert pricing.estimate("opendocrouter", opt(), [], 3).usd is None


def frame(rows):
    return pd.DataFrame(rows, columns=["pdf", "kind", "content", "structure"])


def test_focus_scores_use_tables_forms_figures_and_scan_text():
    f = frame([("a.pdf", "table", 1, 0.5), ("a.pdf", "heading", 1, 0.0), ("b.pdf", "field", 1, 1.0),
               ("s.pdf", "paragraph", 0.5, 0.5), ("s.pdf", "table", 0.5, 0.0)])
    r = recommend.focus_scores(f, {"s.pdf"})
    assert r["ocr"] == 0.5
    assert r["structure on tables, figures, forms"] == 0.625  # table mean 0.25, field 1.0
    assert r["overall"] is not None and "table" in r["weak"]


def model(name, focus, free=True, cost=0.0, runs=2, errors=0, seconds=10):
    return {"name": name, "free": free, "cost": cost, "seconds": seconds, "runs": runs, "expected_runs": 2, "errors": errors,
            "not_applicable": [], "focus": focus, "ocr": None, "overall": focus, "weak": []}


def test_advice_splits_free_and_paid_and_skips_incomplete_runs():
    ms = [model("a", 0.9), model("b", 0.8), model("c", 0.7), model("d", 0.6),
          model("p1", 0.95, False, 1.0), model("p2", 0.5, False, 2.0), model("bad", 0.99, runs=1)]
    adv = recommend.advise(ms)
    assert [m["name"] for m in adv["free"]] == ["a", "b", "c"]
    assert [m["name"] for m in adv["paid"]] == ["p1", "p2"]
    assert adv["excluded"][0][0] == "bad"


def test_advice_warns_on_ties_and_empty_paid():
    adv = recommend.advise([model("a", 0.90), model("b", 0.89)])
    assert any("do not read the order" in n for n in adv["notes"]) and any("No paid model" in n for n in adv["notes"])
