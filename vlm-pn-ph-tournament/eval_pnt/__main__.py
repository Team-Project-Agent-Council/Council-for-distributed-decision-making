"""CLI dispatcher: ``python -m eval <subcommand> ...``.

Subcommands:
  geo, geo-spatial bias metrics (Stage 1, CPU)
  agents, per-agent metrics (Stage 1, CPU)
  funnel, pipeline-funnel + oracle/tournament/agreement diagnostics (Stage 1, CPU)
  heatmap, per-country TPR/FP heatmap, static + interactive (Stage 1, CPU)
  calibration, per-agent Brier/ECE + reliability diagrams (Stage 1, CPU)
  anchoring, tournament anchoring vs. re-ranking (Stage 1, CPU)
  rag-utility, RAG coverage + per-match utility (Stage 1, CPU)
  influence, legacy cross-agent influence (kept for compat; not in `all`)
  judge, LLM-as-judge verdicts (Stage 2, GPU)
  aggregate, fold per-image judge JSONs into a summary
  report, combine everything into a markdown + HTML report
  all, run the full pipeline (Stage 1 + 2 + report)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", type=Path, required=True,
                   help="Path to results_v12_pn/ (parent of per-image dirs)")
    p.add_argument("--gt", type=Path, required=True,
                   help="Path to georc_locations.csv")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (will be created)")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="python -m eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_geo = sub.add_parser("geo", help="Geo-spatial bias metrics")
    _add_common(p_geo)

    p_ag = sub.add_parser("agents", help="Per-agent metrics")
    _add_common(p_ag)

    p_fn = sub.add_parser("funnel", help="Pipeline funnel + oracle diagnostics")
    _add_common(p_fn)

    p_hm = sub.add_parser("heatmap", help="Geographic TPR/FP heatmap")
    _add_common(p_hm)

    p_cal = sub.add_parser("calibration", help="Per-agent Brier/ECE")
    _add_common(p_cal)
    p_anch = sub.add_parser("anchoring", help="Tournament anchoring vs. re-ranking")
    _add_common(p_anch)
    p_rag = sub.add_parser("rag-utility", help="RAG coverage + per-match utility")
    _add_common(p_rag)

    p_dyn = sub.add_parser("dynamics",
                           help="Region narrowing funnel + tournament bracket dynamics")
    _add_common(p_dyn)

    p_inf = sub.add_parser("influence", help="Legacy cross-agent influence")
    _add_common(p_inf)

    p_judge = sub.add_parser("judge", help="LLM-as-judge verdicts")
    _add_common(p_judge)
    p_judge.add_argument("--image-root", type=Path, default=None,
                         help="Where the images live (defaults to image_path "
                              "stored in result.json, set this if paths moved)")
    p_judge.add_argument("--model", default=None,
                         help="Override VLM_MODEL for the judge")
    p_judge.add_argument("--api-base", default=None,
                         help="Override VLM_API_BASE for the judge")
    p_judge.add_argument("--concurrency", type=int, default=1)
    p_judge.add_argument("--limit", type=int, default=None,
                         help="Process only the first N images (smoke test)")

    p_agg = sub.add_parser("aggregate", help="Aggregate per-image judge JSONs")
    p_agg.add_argument("--out", type=Path, required=True)

    p_rep = sub.add_parser("report", help="Markdown + HTML report")
    p_rep.add_argument("--out", type=Path, required=True)

    p_all = sub.add_parser("all", help="Run everything")
    _add_common(p_all)
    p_all.add_argument("--image-root", type=Path, default=None)
    p_all.add_argument("--model", default=None)
    p_all.add_argument("--api-base", default=None)
    p_all.add_argument("--concurrency", type=int, default=1)
    p_all.add_argument("--skip-judge", action="store_true",
                       help="Run only Stage 1 + report (no GPU needed)")
    p_all.add_argument("--no-html", action="store_true",
                       help="Skip rendering the HTML report")

    args = parser.parse_args(argv)

    if args.cmd == "geo":
        from eval_pnt.geo import run as geo_run
        geo_run(args.results, args.gt, args.out)
    elif args.cmd == "agents":
        from eval_pnt.agents import run as agents_run
        agents_run(args.results, args.gt, args.out)
    elif args.cmd == "funnel":
        from eval_pnt.funnel import run as funnel_run
        funnel_run(args.results, args.gt, args.out)
    elif args.cmd == "heatmap":
        from eval_pnt.heatmap import run as heatmap_run
        heatmap_run(args.results, args.gt, args.out)
    elif args.cmd == "calibration":
        from eval_pnt.calibration import run as cal_run
        cal_run(args.results, args.gt, args.out)
    elif args.cmd == "anchoring":
        from eval_pnt.anchoring import run as anch_run
        anch_run(args.results, args.gt, args.out)
    elif args.cmd == "rag-utility":
        from eval_pnt.rag_utility import run as rag_run
        rag_run(args.results, args.gt, args.out)
    elif args.cmd == "dynamics":
        from vlm_council.analyze_rounds_v12 import compute_dynamics
        compute_dynamics(args.results, args.gt, args.out)
    elif args.cmd == "influence":
        from eval_pnt.influence import run as influence_run
        influence_run(args.results, args.gt, args.out)
    elif args.cmd == "judge":
        from eval_pnt.judge import run as judge_run
        judge_run(
            results=args.results,
            gt=args.gt,
            out=args.out,
            image_root=args.image_root,
            model=args.model,
            api_base=args.api_base,
            concurrency=args.concurrency,
            limit=args.limit,
        )
    elif args.cmd == "aggregate":
        from eval_pnt.judge_aggregate import run as agg_run
        agg_run(args.out)
    elif args.cmd == "report":
        from eval_pnt.report import run as rep_run
        rep_run(args.out)
    elif args.cmd == "all":
        from eval_pnt.geo import run as geo_run
        from eval_pnt.agents import run as agents_run
        from eval_pnt.funnel import run as funnel_run
        from eval_pnt.heatmap import run as heatmap_run
        from eval_pnt.calibration import run as cal_run
        from eval_pnt.anchoring import run as anch_run
        from eval_pnt.rag_utility import run as rag_run
        from eval_pnt.report import run as rep_run

        args.out.mkdir(parents=True, exist_ok=True)
        print("=== Stage 1: geo ==="); geo_run(args.results, args.gt, args.out)
        print("=== Stage 1: agents ==="); agents_run(args.results, args.gt, args.out)
        print("=== Stage 1: funnel ==="); funnel_run(args.results, args.gt, args.out)
        print("=== Stage 1: heatmap ==="); heatmap_run(args.results, args.gt, args.out)
        print("=== Stage 1: calibration ==="); cal_run(args.results, args.gt, args.out)
        print("=== Stage 1: anchoring ==="); anch_run(args.results, args.gt, args.out)
        print("=== Stage 1: rag-utility ==="); rag_run(args.results, args.gt, args.out)
        from vlm_council.analyze_rounds_v12 import compute_dynamics
        print("=== Stage 1: dynamics ==="); compute_dynamics(args.results, args.gt, args.out)
        if not args.skip_judge:
            from eval_pnt.judge import run as judge_run
            from eval_pnt.judge_aggregate import run as agg_run
            print("=== Stage 2: judge ===")
            judge_run(
                results=args.results, gt=args.gt, out=args.out,
                image_root=args.image_root, model=args.model,
                api_base=args.api_base, concurrency=args.concurrency, limit=None,
            )
            print("=== Stage 2: aggregate ===")
            agg_run(args.out)
        print("=== Report ===")
        rep_run(args.out)
        if args.no_html:
            html_path = args.out / "report.html"
            if html_path.exists():
                html_path.unlink()
                print("[report] HTML removed (--no-html)")
    else:
        parser.error(f"unknown subcommand: {args.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
