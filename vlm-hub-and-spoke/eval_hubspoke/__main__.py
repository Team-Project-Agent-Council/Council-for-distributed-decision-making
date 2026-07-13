"""CLI dispatcher: python -m eval_hubspoke <subcommand> ...

Subcommands:
  geo        -- geo-spatial bias metrics
  agents     -- per-agent initial + discussion metrics
  judge      -- LLM-as-judge verdicts (Stage 2, requires GPU/API)
  aggregate  -- fold per-image judge JSONs into a summary
  report     -- combine everything into a markdown + HTML report
  all        -- run geo + agents + (judge) + aggregate + report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results", type=Path, required=True,
                   help="Path to results_hub_and_spoke/ (parent of per-image dirs)")
    p.add_argument("--gt", type=Path, required=True,
                   help="Path to georc_locations.csv (ground truth)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (will be created)")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(prog="python -m eval_hubspoke")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_geo = sub.add_parser("geo", help="Geo-spatial bias metrics")
    _add_common(p_geo)

    p_ag = sub.add_parser("agents", help="Per-agent metrics")
    _add_common(p_ag)

    p_hm = sub.add_parser("heatmap", help="Geographic TPR/FP heatmap + world maps")
    _add_common(p_hm)

    p_dyn = sub.add_parser("dynamics",
                           help="Hub and spoke dynamics metrics (rounds, convergence, per-agent updates)")
    _add_common(p_dyn)

    p_judge = sub.add_parser("judge", help="LLM-as-judge verdicts")
    _add_common(p_judge)
    p_judge.add_argument("--image-root", type=Path, default=None,
                         help="Where images live (if paths in result.json moved)")
    p_judge.add_argument("--model", default=None,
                         help="Override VLM_JUDGE_LLM_MODEL")
    p_judge.add_argument("--api-base", default=None,
                         help="Override VLM_JUDGE_LLM_API_BASE")
    p_judge.add_argument("--concurrency", type=int, default=1,
                         help="Number of concurrent judge calls")
    p_judge.add_argument("--limit", type=int, default=None,
                         help="Process only the first N images (smoke test)")
    p_judge.add_argument("--file-list", type=Path, default=None,
                         help="Optional whitelist file (one image_id per line). "
                              "Only those records are judged. Used by the "
                              "parallel short-gpu launcher.")

    p_agg = sub.add_parser("aggregate", help="Aggregate per-image judge JSONs")
    p_agg.add_argument("--out", type=Path, required=True,
                       help="Output directory that contains judge/ subdir")

    p_rep = sub.add_parser("report", help="Markdown + HTML report")
    p_rep.add_argument("--out", type=Path, required=True,
                       help="Output directory containing eval JSONs")

    p_all = sub.add_parser("all", help="Run the full evaluation pipeline")
    _add_common(p_all)
    p_all.add_argument("--image-root", type=Path, default=None)
    p_all.add_argument("--model", default=None)
    p_all.add_argument("--api-base", default=None)
    p_all.add_argument("--concurrency", type=int, default=1)
    p_all.add_argument("--skip-judge", action="store_true",
                       help="Run only Stage 1 (no GPU/LLM needed)")

    args = parser.parse_args(argv)

    if args.cmd == "geo":
        from eval_hubspoke.geo import run as geo_run
        geo_run(args.results, args.gt, args.out)

    elif args.cmd == "agents":
        from eval_hubspoke.agents import run as agents_run
        agents_run(args.results, args.gt, args.out)

    elif args.cmd == "heatmap":
        from eval_hubspoke.heatmap import run as heatmap_run
        heatmap_run(args.results, args.gt, args.out)

    elif args.cmd == "dynamics":
        from vlm_council.analyze_rounds_hub_and_spoke import compute_dynamics
        compute_dynamics(args.results, args.gt, args.out)

    elif args.cmd == "judge":
        from eval_hubspoke.judge import run as judge_run
        judge_run(
            results=args.results,
            gt=args.gt,
            out=args.out,
            image_root=args.image_root,
            model=args.model,
            api_base=args.api_base,
            concurrency=args.concurrency,
            limit=args.limit,
            file_list=args.file_list,
        )

    elif args.cmd == "aggregate":
        from eval_hubspoke.judge_aggregate import run as agg_run
        agg_run(args.out)

    elif args.cmd == "report":
        from eval_hubspoke.report import run as rep_run
        rep_run(args.out)

    elif args.cmd == "all":
        from eval_hubspoke.geo import run as geo_run
        from eval_hubspoke.agents import run as agents_run
        from eval_hubspoke.report import run as rep_run

        args.out.mkdir(parents=True, exist_ok=True)
        print("=== Stage 1: geo ===")
        geo_run(args.results, args.gt, args.out)
        print("=== Stage 1: agents ===")
        agents_run(args.results, args.gt, args.out)
        print("=== Stage 1: heatmap ===")
        from eval_hubspoke.heatmap import run as heatmap_run
        heatmap_run(args.results, args.gt, args.out)

        print("=== Stage 1: dynamics ===")
        from vlm_council.analyze_rounds_hub_and_spoke import compute_dynamics
        compute_dynamics(args.results, args.gt, args.out)

        if not args.skip_judge:
            from eval_hubspoke.judge import run as judge_run
            from eval_hubspoke.judge_aggregate import run as agg_run
            print("=== Stage 2: judge ===")
            judge_run(
                results=args.results,
                gt=args.gt,
                out=args.out,
                image_root=args.image_root,
                model=args.model,
                api_base=args.api_base,
                concurrency=args.concurrency,
                limit=None,
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
