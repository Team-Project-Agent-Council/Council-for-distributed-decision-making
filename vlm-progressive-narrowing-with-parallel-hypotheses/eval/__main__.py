"""CLI dispatcher: ``python -m eval <subcommand> ...``.

Subcommands:
  geo, geo-spatial bias metrics (Stage 1, CPU)
  agents, per-agent metrics (Stage 1, CPU)
  funnel, pipeline funnel / truth-survival analysis (Stage 1, CPU)
  judge, LLM-as-judge verdicts (Stage 2, GPU)
  aggregate, fold per-image judge JSONs into a summary
  report, combine everything into a markdown report
  all, run the full pipeline (Stage 1 + 2 + report)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", type=Path, required=True,
                   help="Path to results/ directory (parent of per-image dirs)")
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

    p_fn = sub.add_parser("funnel", help="Pipeline funnel / truth-survival analysis")
    _add_common(p_fn)

    p_dyn = sub.add_parser("dynamics", help="PN dynamics metrics (Path A/B split, region + country narrowing funnel)")
    _add_common(p_dyn)

    p_hm = sub.add_parser("heatmap", help="Geographic TPR/FP heatmap + world maps")
    _add_common(p_hm)

    p_judge = sub.add_parser("judge", help="LLM-as-judge verdicts")
    _add_common(p_judge)
    p_judge.add_argument("--image-root", type=Path, default=None)
    p_judge.add_argument("--model", default=None)
    p_judge.add_argument("--api-base", default=None)
    p_judge.add_argument("--concurrency", type=int, default=1)
    p_judge.add_argument("--limit", type=int, default=None)
    p_judge.add_argument("--file-list", type=Path, default=None,
                         help="Optional whitelist file (one image_id per line). "
                              "Only those records are judged. Used by the "
                              "parallel short-gpu launcher.")

    p_agg = sub.add_parser("aggregate", help="Aggregate per-image judge JSONs")
    p_agg.add_argument("--out", type=Path, required=True)

    p_rep = sub.add_parser("report", help="Markdown report")
    p_rep.add_argument("--out", type=Path, required=True)

    p_all = sub.add_parser("all", help="Run everything")
    _add_common(p_all)
    p_all.add_argument("--image-root", type=Path, default=None)
    p_all.add_argument("--model", default=None)
    p_all.add_argument("--api-base", default=None)
    p_all.add_argument("--concurrency", type=int, default=1)
    p_all.add_argument("--skip-judge", action="store_true",
                       help="Run only Stage 1 + report (no GPU needed)")

    args = parser.parse_args(argv)

    if args.cmd == "geo":
        from eval.geo import run as geo_run
        geo_run(args.results, args.gt, args.out)
    elif args.cmd == "agents":
        from eval.agents import run as agents_run
        agents_run(args.results, args.gt, args.out)
    elif args.cmd == "funnel":
        from eval.funnel import run as funnel_run
        funnel_run(args.results, args.gt, args.out)
    elif args.cmd == "dynamics":
        from vlm_council.analyze_rounds_progressive_narrowing import compute_dynamics
        compute_dynamics(args.results, args.gt, args.out)
    elif args.cmd == "heatmap":
        from eval.heatmap import run as heatmap_run
        heatmap_run(args.results, args.gt, args.out)
    elif args.cmd == "judge":
        from eval.judge import run as judge_run
        judge_run(
            results=args.results, gt=args.gt, out=args.out,
            image_root=args.image_root, model=args.model,
            api_base=args.api_base, concurrency=args.concurrency,
            limit=args.limit, file_list=args.file_list,
        )
    elif args.cmd == "aggregate":
        from eval.judge_aggregate import run as agg_run
        agg_run(args.out)
    elif args.cmd == "report":
        from eval.report import run as rep_run
        rep_run(args.out)
    elif args.cmd == "all":
        from eval.geo import run as geo_run
        from eval.agents import run as agents_run
        from eval.funnel import run as funnel_run
        from eval.heatmap import run as heatmap_run
        from eval.report import run as rep_run

        args.out.mkdir(parents=True, exist_ok=True)
        print("=== Stage 1: geo ==="); geo_run(args.results, args.gt, args.out)
        print("=== Stage 1: agents ==="); agents_run(args.results, args.gt, args.out)
        print("=== Stage 1: funnel ==="); funnel_run(args.results, args.gt, args.out)
        print("=== Stage 1: heatmap ==="); heatmap_run(args.results, args.gt, args.out)
        from vlm_council.analyze_rounds_progressive_narrowing import compute_dynamics
        print("=== Stage 1: dynamics ==="); compute_dynamics(args.results, args.gt, args.out)
        if not args.skip_judge:
            from eval.judge import run as judge_run
            from eval.judge_aggregate import run as agg_run
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
    else:
        parser.error(f"unknown subcommand: {args.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
