"""The questions behind each score, the LLM judge (with a fake model), the comparison, and re-evaluation."""

import json
import re
import subprocess
import sys

import pandas as pd
import pytest

from pdf_bench import analysis, judge, locate, questions, selection, view
from pdf_bench.elements import load_gt
from pdf_bench.oracle import gt_to_markdown
from pdf_bench.score import score
from pdf_bench.synth import generate_structured


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    d = tmp_path_factory.mktemp("judge_docs")
    generate_structured(d)
    return d


def scored(docs, name="forms", text=None):
    pdf = docs / f"{name}.pdf"
    gt = locate.annotate(load_gt(pdf.with_suffix(".gt.json")), pdf)
    text = gt_to_markdown(gt) if text is None else text
    return pdf, score(gt, [text], parser="x").frame(), [text]


class FakeModel:
    """Answers every question with structure 1, except those whose number is in ``wrong`` (structure 0)."""

    def __init__(self, wrong=()):
        self.wrong, self.calls = set(wrong), 0

    def __call__(self, settings, system, user):
        self.calls += 1
        if "CASES:" in user:  # the arbitration pass: the judge sees both scores; it sides with the rules
            self.cases = getattr(self, "cases", []) + [user]
            ids = [int(i) for i in re.findall(r"^Case (\d+) \(", user, re.M)]
            return json.dumps([{"id": i, "ruling": "rules", "correct": 1, "explanation": "rules checked", "rule_fix": ""} for i in ids]), \
                {"input_tokens": 10, "output_tokens": 5}
        ids = [int(i) for i in re.findall(r"^Question (\d+) \(", user, re.M)]
        items = [{"id": i, "content": 1, "structure": 0 if i in self.wrong else 1, "reason": f"seen {i}"} for i in ids]
        return "```json\n" + json.dumps(items) + "\n```", {"input_tokens": 100, "output_tokens": 20}


def test_every_scored_row_is_a_question(docs):
    for kind in ("forms", "lists", "table_merged", "table_multipage", "footnotes", "figures", "constat", "running", "headings"):
        _, f, _ = scored(docs, kind)
        for r in f.itertuples():
            assert r.kind in questions.RUBRIC, r.kind
            if r.kind != "page":
                assert r.spec, f"{kind}: no question for the {r.kind} row {r.expected!r}"


def test_specs_carry_what_the_judge_needs(docs):
    _, f, _ = scored(docs, "table_merged")
    spec = f[f.kind == "table"].iloc[0].spec
    assert "row 1, col 1" in spec and "spans" in spec
    _, f, _ = scored(docs, "forms")
    assert any("ticked" in s or "empty" in s for s in f[f.kind == "checkbox"].spec)
    _, f, _ = scored(docs, "table_multipage")
    assert "ONE table" in f[f.kind == "table_continuity"].iloc[0].spec


def test_parse_is_forgiving():
    out = judge.parse('Sure:\n```json\n[{"id": 1, "content": 2, "structure": "0.5", "reason": "x"}, {"id": 9}, {"id": 2, "structure": null}]\n```', [1, 2])
    assert out[1] == {"content": 1.0, "structure": 0.5, "reason": "x"}
    assert out[2]["structure"] is None and 9 not in out
    assert judge.parse("not json", [1]) == {}


def test_judge_caches_and_reuses(docs, tmp_path, monkeypatch):
    _, f, pages = scored(docs, "forms")
    fake = FakeModel(wrong={1})
    monkeypatch.setattr(judge, "_call", fake)
    tally: dict = {}
    g = judge.judge_frame(f, pages, cache_dir=tmp_path, tally=tally)
    assert tally["calls"] == 1 and fake.calls == 1
    assert g["judge_structure"].iloc[0] == 0 and g["judge_structure"].iloc[1] == 1 and g["judge_reason"].iloc[1] == "seen 2"
    # second time: from disk, no call
    tally2: dict = {}
    again = judge.judge_frame(f, pages, cache_dir=tmp_path, tally=tally2)
    assert fake.calls == 1 and tally2["cached"] == 1 and tally2["calls"] == 0
    pd.testing.assert_frame_equal(g, again)
    # call=False on an empty cache: nothing sent, rows left empty
    empty = judge.judge_frame(f, pages, cache_dir=tmp_path / "other", call=False)
    assert fake.calls == 1 and empty["judge_structure"].isna().all()
    # force asks again; another output is another question
    judge.judge_frame(f, pages, cache_dir=tmp_path, force=True)
    assert fake.calls == 2
    judge.judge_frame(f, [pages[0] + "\n\nmore"], cache_dir=tmp_path)
    assert fake.calls == 3


def test_a_failed_call_is_reported_not_raised(docs, tmp_path, monkeypatch):
    _, f, pages = scored(docs, "forms")

    def boom(*a):
        raise RuntimeError("rate limit")

    monkeypatch.setattr(judge, "_call", boom)
    tally: dict = {}
    g = judge.judge_frame(f, pages, cache_dir=tmp_path, tally=tally)
    assert tally["errors"] == 1 and g["judge_structure"].isna().all() and "rate limit" in g["judge_reason"].iloc[0]


def test_not_applicable_structure_is_not_judged(docs, tmp_path, monkeypatch):
    pdf = docs / "links.pdf"
    monkeypatch.setattr(judge, "_call", FakeModel())
    monkeypatch.setattr(judge, "DEFAULT_DIR", tmp_path)
    f = view.frame("claude", "claude", pdf, [gt_to_markdown(load_gt(pdf.with_suffix(".gt.json")))])
    assert f.loc[f.kind == "link", "structure"].isna().all()
    g = view.judged(f, "claude", [gt_to_markdown(load_gt(pdf.with_suffix(".gt.json")))], cache_dir=tmp_path)
    assert g.loc[g.kind == "link", "judge_structure"].isna().all()
    assert g.loc[g.kind == "link", "judge_content"].notna().all()  # the text itself is still judged


def test_agreement_tools():
    f = pd.DataFrame({"model": "m", "kind": ["a", "a", "b", "b", "b", "b"],
                      "structure": [1, 1, 0, 1, 0, 1], "judge_structure": [1, 0, 0, 1, 1, 1]})
    ag = judge.agreement(f)
    assert ag.loc["a", "questions"] == 2 and ag.loc["a", "agree %"] == 50 and ag.loc["a", "judge stricter"] == 1
    assert ag.loc["b", "rules stricter"] == 1
    d = judge.disagreements(f)
    assert len(d) == 2 and set(d["kind"]) == {"a", "b"}
    assert judge.cohen_kappa(f) == pytest.approx(0.25, abs=0.01)
    assert judge.cohen_kappa(f.head(3)) is None


def test_analysis_files_with_the_judge(docs, tmp_path, monkeypatch):
    from pdf_bench import results

    monkeypatch.chdir(tmp_path)
    d = tmp_path / "data" / "set"
    generate_structured(d, only=["forms"])
    sel = selection.Selection(parsers=["pymupdf4llm"], docs_dir="data/set")
    results.run("pymupdf4llm", d / "forms.pdf")
    fake = FakeModel(wrong={1})
    monkeypatch.setattr(judge, "_call", fake)
    monkeypatch.setattr(judge, "DEFAULT_DIR", tmp_path / "results" / "judge")
    # cached mode before any call: nothing is sent
    data = analysis.collect(sel, judge_mode="cached")
    assert fake.calls == 0 and data["judge"] and data["runs"][0]["rows"][0]["judge_structure"] is None
    md, js = analysis.write(sel, judge_mode="run")
    assert fake.calls >= 1
    text = md.read_text(encoding="utf-8")
    assert "## Rules against the LLM judge" in text and "judge_structure" in text
    row = json.loads(js.read_text(encoding="utf-8"))["runs"][0]["rows"][0]
    assert row["judge_structure"] == 0 and row["question"]
    # the judge's own report: coverage, agreement, and the disagreement (the first question was answered 0 by the judge)
    jmd, jjs = analysis.write_judge(sel)
    rep = jmd.read_text(encoding="utf-8")
    assert jmd.name == "judge_analysis.md" and jjs.exists()
    assert "## Per model" in rep and "## Disagreements" in rep and "judge: seen" not in rep and "expected:" in rep
    assert "Generated" in rep and "Judge: `anthropic:" in rep
    # off: the files have no judge at all
    assert "judge_structure" not in analysis.write(sel)[0].read_text(encoding="utf-8")


def test_reevaluate_reloads_the_evaluation_modules():
    code = ("from pdf_bench import judge, rerun, score; before = score.score; done = rerun.reevaluate(); "
            "import pdf_bench.score as s; assert 'score' in done and 'judge' in done and s.score is not None; print(','.join(done))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr


APP_SCRIPT = r'''
import os, sys
from pathlib import Path
from streamlit.testing.v1 import AppTest
from pdf_bench import results, selection
from pdf_bench.synth import generate_structured

work = Path(sys.argv[1])
os.chdir(work)
d = work / "data" / "set"
generate_structured(d, only=["forms", "lists"])
selection.Selection(parsers=["pymupdf4llm"], docs_dir="data/set").save(work / "b.json")
for name in ("forms.pdf", "lists.pdf"):
    results.run("pymupdf4llm", d / name)
page = str(Path(selection.__file__).with_name("page_compare.py"))
at = AppTest.from_file(page, default_timeout=120)
at.session_state["selection_file"] = str(work / "b.json")
at.run()
assert not at.exception, at.exception
# the app no longer has a judge section; the header says which evaluation produced the numbers
assert not any("judge" in str(x.value).lower() for x in at.subheader)
assert not any(b.label == "Run the judge" for b in at.button)
assert any(c.value.startswith("Evaluation `") for c in at.caption), [c.value[:40] for c in at.caption]
# editing the evaluation, then pressing the button: reloaded, nothing parsed
next(b for b in at.button if b.label == "Rerun evaluation").click()
at.run()
assert not at.exception, at.exception
assert any("Evaluation reloaded" in s.value for s in at.success), [s.value for s in at.success]
print("ok")
'''


def test_compare_page_has_no_judge_and_rerun_works(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(APP_SCRIPT, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    out = subprocess.run([sys.executable, str(script), str(work)], capture_output=True, text=True, check=False, timeout=280)
    assert out.returncode == 0 and "ok" in out.stdout, out.stdout[-800:] + out.stderr[-2500:]


def test_temperature_refusal_is_retried_and_errors_are_kept():
    seen = []

    def send(**kw):
        seen.append(kw)
        if "temperature" in kw:
            raise ValueError("`temperature` is deprecated for this model")
        return "ok"

    assert judge._without_temperature(send) == "ok" and seen == [{"temperature": 0}, {}]
    with pytest.raises(ValueError):
        judge._without_temperature(lambda **kw: (_ for _ in ()).throw(ValueError("bad model")))


def test_the_judge_rules_on_its_disagreements(docs, tmp_path, monkeypatch):
    _, f, pages = scored(docs, "forms")
    fake = FakeModel(wrong={1})  # the judge says 0 on question 1, the rules say 1
    monkeypatch.setattr(judge, "_call", fake)
    tally: dict = {}
    g = judge.judge_frame(f, pages, cache_dir=tmp_path, tally=tally, arbitrate=True)
    assert g["arb_ruling"].iloc[0] == "rules" and g["arb_reason"].iloc[0] == "rules checked" and g["arb_correct"].iloc[0] == 1
    assert (g["arb_ruling"].iloc[1:] == "").all()  # no disagreement, no ruling
    case = fake.cases[0]
    assert "RULES: structure = 1.00" in case and "JUDGE: structure = 0.00 (seen 1)" in case and "disputed: structure" in case
    # nothing asked again from the cache; a cache-only read shows the ruling without a call
    n = fake.calls
    again = judge.judge_frame(f, pages, cache_dir=tmp_path, call=False, arbitrate=True)
    assert fake.calls == n and again["arb_ruling"].iloc[0] == "rules"
    # the blind pass never sees the rules
    first = judge.prompt(pages, [(1, "x", "spec")])
    assert "RULES" not in first and "JUDGE" not in first
    r = judge.rulings(g)
    assert r.loc[g["kind"].iloc[0], "rules right"] == 1


def test_a_content_disagreement_counts(docs):
    f = pd.DataFrame({"model": "m", "kind": ["a", "a"], "content": [1.0, 1.0], "judge_content": [0.0, 1.0],
                      "structure": [1.0, 1.0], "judge_structure": [1.0, 1.0], "expected": "e", "detail": "", "judge_reason": ""})
    d = judge.disagreements(f)
    assert len(d) == 1 and d["disputed"].iloc[0] == "content"
