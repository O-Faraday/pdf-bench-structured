"""Command line: ``pdf-bench synth|parsers|analysis|evaluate|app``."""

from __future__ import annotations

import argparse
import sys


def cmd_synth(args: argparse.Namespace) -> int:
    from .elements import load_gt
    from .synth import KINDS, check_gt, generate_structured, plan

    only = [k.strip() for k in args.only.split(",") if k.strip()] if args.only else None
    try:
        wanted = plan(args.count, args.seed, only)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"writing {len(wanted)} PDF(s) to {args.dir} ({len(only or KINDS)} kinds, seed {args.seed})")
    bad = 0
    for pdf in generate_structured(args.dir, count=args.count, seed=args.seed, only=only):
        gt = load_gt(pdf.with_suffix(".gt.json"))
        problems = check_gt(pdf, gt)
        bad += len(problems)
        kinds = ", ".join(f"{n} {k}" for k, n in gt.counts().items())
        print(f"{pdf.name:26} {kinds}" + (f"  !! {problems[0]}" if problems else ""))
    return 1 if bad else 0


def cmd_parsers(args: argparse.Namespace) -> int:
    from .parsers import CHECKED, FAMILIES, REGISTRY

    print(f"structured parsers (catalogue checked {CHECKED})")
    for family, title in FAMILIES.items():
        rows = [p for p in REGISTRY.values() if p.family == family]
        if not rows:
            continue
        print(f"\n{title}")
        for p in rows:
            state = p.status() or "ready"
            flags = ("hosted " if p.hosted else "") + ("GPU " if p.gpu else "")
            print(f"  {p.name:20} {flags:12} {state[:90]}")
    return 0


def cmd_analysis(args: argparse.Namespace) -> int:
    from . import analysis, config, selection

    config.load_env()
    sel = selection.load(args.selection)
    sel.apply()
    mode = "run" if args.judge else "cached" if args.judge_cached else "off"
    if mode == "run" and (why := _judge_problem()):
        print(f"the judge cannot be called: {why}", file=sys.stderr)
        return 2
    md, js = analysis.write(sel, args.out, judge_mode=mode)
    print(f"wrote {md}\nwrote {js}")
    if mode != "off":  # the verdicts are on disk now: the judge's own report reads them
        jmd, jjs = analysis.write_judge(sel, args.out)
        print(f"wrote {jmd}\nwrote {jjs}")
    return 0


def cmd_add_odr(args: argparse.Namespace) -> int:
    """Add OpenDocRouter models to the saved selection as named variants ``odr:<model>``."""
    from . import selection
    from .parsers.opendocrouter_util import label

    sel = selection.load(args.selection)
    for model in args.models:
        sel.variants[label(model)] = {"parser": "opendocrouter", "options": {"model": model}}
        print(f"added {label(model)}  ({model})")
    sel.allow_hosted = True  # asking for a hosted model by name is the confirmation the app would ask for
    print(f"saved {sel.save(args.selection)}; run it from the app (Run), it needs OPEN_DOC_ROUTER_API_KEY in .env")
    return 0


def _judge_problem() -> str:
    from . import judge

    try:
        return judge.missing_settings()
    except ValueError as exc:
        return str(exc)


def cmd_app(args: argparse.Namespace) -> int:
    import subprocess
    from pathlib import Path

    app = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *args.streamlit_args])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdf-bench", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write synthetic PDFs with an element-level ground truth")
    p.add_argument("dir", nargs="?", default="data/structured")
    p.add_argument("--count", type=int, metavar="N", help="number of PDFs to create (default: one per kind)")
    p.add_argument("--seed", type=int, default=0, help="first seed; seed 0 is the reference document (default 0)")
    p.add_argument("--only", metavar="KINDS", help="comma-separated kinds to use, e.g. constat,table_nested (see docs/STRUCTURED.md)")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("parsers", help="list the structured parsers and whether each can run here")
    p.set_defaults(fn=cmd_parsers)

    p = sub.add_parser("analysis", help="write the analysis files (scores, evidence, outputs) from the saved answers")
    p.add_argument("--selection", default="benchmark.json", help="the saved selection (default benchmark.json)")
    p.add_argument("--out", default="analysis", help="output folder (default analysis/)")
    p.add_argument("--judge", action="store_true", help="also ask the LLM judge the same questions (costs tokens; verdicts are saved in results/judge)")
    p.add_argument("--judge-cached", action="store_true", help="add the judge's saved verdicts without calling it")
    p.set_defaults(fn=cmd_analysis)
    p = sub.add_parser("evaluate", help="same as analysis: score the saved answers again, no parsing (after editing the evaluation)")
    p.add_argument("--selection", default="benchmark.json")
    p.add_argument("--out", default="analysis")
    p.add_argument("--judge", action="store_true", help="also ask the LLM judge (costs tokens)")
    p.add_argument("--judge-cached", action="store_true", help="add the judge's saved verdicts without calling it")
    p.set_defaults(fn=cmd_analysis)

    p = sub.add_parser("add-odr", help="add OpenDocRouter models (e.g. openai/gpt-5.6-terra) to the saved selection")
    p.add_argument("models", nargs="+", metavar="MODEL_ID")
    p.add_argument("--selection", default="benchmark.json")
    p.set_defaults(fn=cmd_add_odr)

    p = sub.add_parser("app", help="open the app: create documents, choose the parsers")
    p.add_argument("streamlit_args", nargs="*", help="passed to streamlit, e.g. --server.port 8600")
    p.set_defaults(fn=cmd_app)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
